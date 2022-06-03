"""
Validation script
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from dataloaders.dev_customized_med import med_fewshot_val

from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk
import SimpleITK as sitk
from util.metric import Metric

def get_unlbl_pseudo_lbls_sep(model,sup_dataset,unlbl_ids,_config,_log, val_mode = False):
    """
    This function will make predictions for each class and store them in separate channels
    out_size : ( H, W, max_class, C )
    
    """
    model.eval()
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')
    
    # Predict all classes
    if val_mode:
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    else:
        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all']
    te_transforms = None

    norm_func = get_normalize_op(modality = 'MR', fids = None)
    data_dir = _config['path'][baseset_name]['data_dir']

    ## te_parent must contain only the labeled images (To be used as support)
    ## te_dataset must return only the query images from unlbl_ids
    te_dataset, te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        base_dir=data_dir,
        tr_pids=unlbl_ids,
        transforms=te_transforms,
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func
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

    if val_mode:
        _log.info('###### Set validation nodes ######')
        mar_val_metric_node = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load))
        mar_val_metric_node.reset()

    _log.info('###### Starting pseudo-label estimation ######')

    with torch.no_grad():
        # Index into the buffer and keep adding preds
        _lb_buffer = {} # indexed by scan

        for curr_lb in test_labels:
            te_dataset.set_curr_cls(curr_lb)
            support_batched = sup_dataset.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            support_images = [[shot.cuda() for shot in way]
                                for way in support_batched['support_images']] # way x part x [shot x C x H x W]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]

            curr_scan_count = -1 # counting for current scan
            
            for sample_batched in testloader:
                _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                # if _scan_id in sup_dataset.potential_support_sid: # skip the support scan, don't include that to query
                #     continue
                if sample_batched["is_start"]:
                    ii = 0
                    curr_scan_count += 1
                    _scan_id = sample_batched["scan_id"][0]
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
                _pred[...,ii] = query_pred.copy()

                # If we are in validation mode, no need to save the predictions
                if val_mode:
                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and \
                        (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        
                        mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                    else:
                        pass

                else:
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

        if val_mode:
            m_classDice,_, m_meanDice,_, m_rawDice = mar_val_metric_node.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw = True)
            _log.info(f'###### Mean validation dice score = {m_meanDice} ######')

        else:
            ### save results
            for _scan_id, _pred in _lb_buffer.items():
                # _pred *= float(curr_lb)
                itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
                fid = os.path.join(data_dir, f'label_{_scan_id}.nii.gz')
                sitk.WriteImage(itk_pred, fid, True)
                _log.info(f'###### Pseudo-label for {_scan_id} has been saved ######')


    del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

    print('###### Finished estimating pseudo-labels! ######')
    
    print("============ ============")
    if val_mode:
        return m_meanDice

    return 1
