"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import numpy as np

from models.contrastive_fewshot import FewShotSeg, ContrastiveBranch
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.dataset_utils import DATASET_INFO, update_class_slice_index
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/summary', exist_ok=True)
        writer = SummaryWriter(f'{_run.observers[0].dir}/summary')

        """for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')"""
        
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create models ######')
    # Contrastive loss branch
    model_cont = ContrastiveBranch(cfg=_config['model'])
    model_cont = model_cont.cuda()
    model_cont.train()

    # Segmentation branch
    model = FewShotSeg(shared_encoder=model_cont.unet.layers[0],pretrained_path=_config['reload_model_path'], cfg=_config['model'])
    model = model.cuda()
    model.train()

    
    _log.info('###### Load data ######')
    
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS':
        baseset_name = 'SABS'
    elif data_name == 'C0':
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')
    
    update_class_slice_index(baseset_name, int(_config["min_fg_data"]))
    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    # These are the labeled sets - rest are treated as unlabeled
    tr_pids = DATASET_INFO[baseset_name]['LBL_PIDS']
    tr_paired,tr_parent = med_fewshot(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        # idx_split = _config['eval_fold'],
        tr_pids=tr_pids,
        mode='train',
        scan_per_load = _config['scan_per_load'],
        transforms = None,
        exclude_list=_config["exclude_cls_list"],
        min_fg=str(_config["min_fg_data"]),
        act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]], # This needs to be only the train set we are using (Eg. Liver,Spleen)
        n_ways=1,
        n_shots=1,
        max_iters_per_load=_config['max_iters_per_load']
        )

    
    ### dataloaders
    trainloader = DataLoader(
        tr_paired,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    pids_used = set()
    labels_train = set()

    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            pids_used.add(sample_batched['supp_scan_z_ids'][0][0][0])
            pids_used.add(sample_batched['query_scan_z_ids'][0][0][0])
            labels_train.add(sample_batched['class_ids'][0].cpu().numpy()[0])
            
            # Prepare input
            i_iter += 1

            support_images = [[shot.cuda() for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
           
            query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)


            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss
            if (i_iter + 1) % _config['print_interval'] == 0:

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']
                writer.add_scalar('train/total_loss', loss, i_iter + 1)
                writer.add_scalar('train/align_loss', align_loss, i_iter + 1)
                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss},')                


    _log.info('###### PIDs used in training : {} ######'.format(pids_used))
    _log.info('###### Labels used in training : {} ######'.format(labels_train))
    _log.info('###### Taking snapshot ######')
    torch.save(model.state_dict(),
                os.path.join(f'{_run.observers[0].dir}/snapshots', f'ft_final.pth'))
    
    return 1 # Finish up