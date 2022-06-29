"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np

from models.contrastive_fewshot import FewShotSeg, ContrastiveBranch

from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op, get_CT_norm_op,update_class_slice_index
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric

from config_ssl_upload import ex

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid
import re
from util.utils import CircularList
import glob

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        """for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')"""
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    _log.info('###### Create models ######')
    # Contrastive loss branch
    model_cont = ContrastiveBranch(cfg=_config['model'])
    model_cont = model_cont.cuda()

    # Segmentation branch
    model = FewShotSeg(shared_encoder=model_cont.unet.layers[0],pretrained_path=_config['reload_model_path'], cfg=_config['model'])
    model = model.cuda()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0':
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    ## Getting the curr fold validation PIDs
    lbl_pids = DATASET_INFO[baseset_name]['LBL_PIDS']
    img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(_config['path'][data_name]['data_dir'] + "/image_*.nii.gz") ]
    available_pids = CircularList(sorted(list(set(img_pids) - set(lbl_pids)),key=lambda x: int(x)))
    if (len(lbl_pids) == 1):
        sep = DATASET_INFO[baseset_name]['_SEP1']
    elif len(lbl_pids) == 4:
        sep = DATASET_INFO[baseset_name]['_SEP']
    idx_split = _config['eval_fold']
    val_pids  = available_pids[sep[idx_split]: sep[idx_split + 1]]
    unlbl_pids = [ii for ii in available_pids if ii not in val_pids]
    tr_pids_all = lbl_pids + unlbl_pids

    if baseset_name == 'SABS': # for CT we need to know statistics of the entire training data
        norm_func = get_CT_norm_op(which_dataset=baseset_name, base_dir=_config['path'][baseset_name]['data_dir'], tr_pids=tr_pids_all)
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)

    print(f"Validation PIDs : {val_pids}")
    update_class_slice_index(baseset_name,min_fg=_config['min_fg_data'])
    
    te_dataset, _ = med_fewshot_val(
        dataset_name = baseset_name,
        tr_pids=val_pids,
        base_dir=_config['path'][baseset_name]['data_dir'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func,
        min_fg=str(_config['min_fg_data'])
    )
    
    _,te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        tr_pids=lbl_pids,
        base_dir=_config['path'][baseset_name]['data_dir'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func,
        min_fg=str(_config['min_fg_data'])
    )

    ### dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )
    _log.info('###### Set validation nodes ######')
    mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load))

    _log.info('###### Starting validation ######')
    model.eval()
    mar_val_metric_node.reset()
    with torch.no_grad():
        # Index into the buffer and keep adding preds
        save_pred_buffer = {} # indexed by class

        for curr_lb in test_labels:
            te_dataset.set_curr_cls(curr_lb)
            support_batched = te_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            support_images = [[shot.cuda() for shot in way]
                                for way in support_batched['support_images']] # way x part x [shot x C x H x W]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]

            curr_scan_count = -1 # counting for current scan
            _lb_buffer = {} # indexed by scan

            last_qpart = 0 # used as indicator for adding result to buffer

            for sample_batched in testloader:

                _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                    continue
                if sample_batched["is_start"]:
                    ii = 0
                    curr_scan_count += 1
                    _scan_id = sample_batched["scan_id"][0]
                    
                    """outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                    outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                    _pred = np.zeros( outsize )
                    _pred.fill(np.nan)"""
                    
                    outsize_orig = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                    outsize = (256, 256, outsize_orig[0]) 
                    _pred = np.zeros( outsize )
                    _pred.fill(np.nan)
                    overall_outsize = (outsize_orig[0],max_label+1,256,256) # (Channels,num_class,H,W) : Incl. bg as placeholder
                    overall_pred = np.zeros(overall_outsize)

                q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                query_images = [sample_batched['image'].cuda()]
                query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                # [way, [part, [shot x C x H x W]]] ->
                sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                query_pred, _, _, assign_mats = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                query_pred = np.array(query_pred.argmax(dim=1)[0].cpu())
                _pred[..., ii] = query_pred.copy()

                if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                    mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                else:
                    pass

                ii += 1
                # now check data format
                if sample_batched["is_end"]:
                    if _config['dataset'] != 'C0':
                        if not _scan_id in _lb_buffer.keys():
                                overall_pred[:,curr_lb,:,:] = _pred.transpose(2,0,1)
                                _lb_buffer[_scan_id] = overall_pred # (H, W, num_class, Z) -> (Z, numclass, H, W)
                        else:
                            ## Keep labels in different channels to avoid confusion at boundaries
                            _lb_buffer[_scan_id][:,curr_lb,:,:] = _pred.transpose(2,0,1)
                    else:
                        _lb_buffer[_scan_id] = _pred

            save_pred_buffer[str(curr_lb)] = _lb_buffer

        ### save results - Commented for now to save storage
        """
        for curr_lb, _preds in save_pred_buffer.items():
            for _scan_id, _pred in _preds.items():
                _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
                fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### {fid} has been saved ######')
        """

        del save_pred_buffer

    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

    # compute dice scores by scan
    m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)

    m_classPrec,_, m_meanPrec,_,  m_classRec,_, m_meanRec,_, m_rawPrec, m_rawRec = mar_val_metric_node.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw = True)

    mar_val_metric_node.reset() # reset this calculation node

    # write validation result to log file
    _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
    _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
    _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

    _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
    _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

    _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
    _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

    _log.info(f'mar_val batches classDice: {m_classDice}')
    _log.info(f'mar_val batches meanDice: {m_meanDice}')

    _log.info(f'mar_val batches classPrec: {m_classPrec}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

    _log.info(f'mar_val batches classRec: {m_classRec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')

    print("============ ============")

    _log.info(f'End of validation')
    return 1