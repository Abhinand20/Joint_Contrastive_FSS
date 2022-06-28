"""
Added Inter-contrastive loss 
FIXME:
    - Cases when images picked for inter-contrastive calc. don't share any positive labels 
"""

# Intra-class approach 1
# Initially, get the initial pseudo-labels for unlbl images
# Sample a single image (from the entire dataset) at first -> pass to segmentation branch, get segmentation loss
# Create 2 augmentations for that single image & make the same classes in both augmentations similar (positive), and negative classes to be different

import os
import shutil
import math
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from models.contrastive_fewshot import FewShotSeg, ContrastiveBranch
from losses import local_cont_loss
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.dataset_utils import DATASET_INFO,update_class_slice_index, get_CT_norm_op, get_normalize_op
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric
from util.utils import calc_pos_neg_index
from config_ssl_upload import ex
from predict import get_unlbl_pseudo_lbls_sep
import glob
import re
from util.utils import CircularList

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/summary', exist_ok=True)
        writer = SummaryWriter(f'{_run.observers[0].dir}/summary')

        ## For debugging puposes
        """for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')"""
            
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    
    lambda_local = 0.1
    data_name = _config['dataset']
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
    model_seg = FewShotSeg(shared_encoder=model_cont.unet.layers[0],pretrained_path=_config['reload_model_path'], cfg=_config['model'])
    model_seg = model_seg.cuda()
    model_seg.train() # Is this needed right now? - Yes, we are updating the conv layers

    
    # Get a concatenated list of parameters to update
    seg_params = []
    for name,param in model_seg.named_parameters():
        if "shared" not in name:
            seg_params.append(param)

    all_params = list(model_cont.parameters()) + seg_params
    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(all_params, **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight) # Not needed?



    _log.info('###### Load data ######')
    ### Training set

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

    ## Transforms for data augmentation (Geometric + random intensity)
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    ## To set up the labeled and unlabeled images (k-Fold CV)
    # These are the labeled sets - rest are treated as unlabeled
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
    print("Validation PIDs - {}".format(val_pids))
    print("Training PIDs - {}".format(tr_pids_all))

    if baseset_name == 'SABS': # for CT we need to know statistics of the entire training data
        norm_func = get_CT_norm_op(which_dataset=baseset_name, base_dir=_config['path'][baseset_name]['data_dir'], tr_pids=tr_pids_all)
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)
    update_class_slice_index(baseset_name, int(_config["min_fg_data"]))
    ### DATSET SHOULD CONTAIN ALL TRAINING IMAGES (NOT JUST LABELED) WHILE UNSUPERVISED TRAINING
    tr_paired,act_train_parent = med_fewshot(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        # idx_split = _config['eval_fold'],
        tr_pids=tr_pids_all,
        unlbl_pids=unlbl_pids,
        test_labels=test_labels,
        mode='train',
        scan_per_load = _config['scan_per_load'],
        transforms = None,
        min_fg=str(_config["min_fg_data"]),
        # act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]], # This needs to be only the set we are training on (Eg. Liver,Spleen)
        act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'], # Can train on all pseudo-labels (Skip when query is a labeled image with unseen label - This means query will have access to GT unseen label)
        n_ways=1,
        n_shots=1,
        max_iters_per_load=_config['max_iters_per_load'],
        norm_func=norm_func
    )
    
    ### While estimating pseudo-labels just use the labeled support images (Can be experimented with later on)
    _,tr_parent = med_fewshot(
        dataset_name=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        # idx_split = _config['eval_fold'],
        tr_pids=lbl_pids,
        mode='train',
        scan_per_load = _config['scan_per_load'],
        transforms = None,
        min_fg=str(_config["min_fg_data"]),
        act_labels=DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'], # This has to contain all classes, as it is being used for providing support images to unlabeled query images (We predict all labels)
        n_ways=1,
        n_shots=1,
        max_iters_per_load=_config['max_iters_per_load'],
        norm_func=norm_func
    )

    # model_seg gets set to eval here    
    # print("### Generating initial pseudo-labels ####")
    get_unlbl_pseudo_lbls_sep(model=model_seg,sup_dataset = tr_parent,unlbl_ids=unlbl_pids,norm_func = norm_func, _config=_config,_log=_log)
    model_seg.train()
    update_class_slice_index(baseset_name, int(_config["min_fg_data"]))

    ### dataloaders
    trainloader = DataLoader(
        tr_paired,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    print("Train slices : {}".format(tr_pids_all))
    print("Slices per epoch : {}".format(len(trainloader)))

    # total number of steps
    overall_epoch_i = 0 
    ## 5 times pseudo-labels will be updated + 1 initial update
    max_iters = _config['total_iters']
    log_loss = {'total_loss': 0, 'seg_loss': 0, 'cont_loss':0}
    pids_used = set()
    labels_train = set()
    supp_cls_pid_idx = defaultdict(set) # To see which PIDs were used for the support classes during training
    max_val_dsc = 1e-16

    _log.info('###### Training ######')

    for iter_no in range(max_iters+1):
        
        if(iter_no!=0):
            print("Re-Estimate labels")
            get_unlbl_pseudo_lbls_sep(model=model_seg,sup_dataset = tr_parent,norm_func = norm_func, unlbl_ids=unlbl_pids,_config=_config,_log=_log)  
            model_seg.train()
            update_class_slice_index(baseset_name, int(_config["min_fg_data"]))
            _log.info('###### Support PIDs for each class : {} ######'.format(supp_cls_pid_idx))
            
        ## If we finished the final cycle, save the best model and return
        if(iter_no==max_iters):
            _log.info('###### PIDs used in training : {} ######'.format(pids_used))
            _log.info('###### Labels used in training : {} ######'.format(labels_train))

            curr_val_dsc = get_unlbl_pseudo_lbls_sep(model=model_seg,sup_dataset = tr_parent,norm_func = norm_func,unlbl_ids=val_pids,_config=_config,_log=_log,val_mode=True)  
            model_seg.train()
            writer.add_scalar('train/val_dsc', curr_val_dsc, overall_epoch_i + 1)

            if curr_val_dsc > max_val_dsc:
                max_val_dsc = curr_val_dsc
                _log.info('###### Taking snapshot ######')
                torch.save(model_seg.state_dict(),
                        os.path.join(f'{_run.observers[0].dir}/snapshots', f'seg_best.pth'))
            
            writer.add_scalar('train/best_val_dsc', max_val_dsc, overall_epoch_i + 1)
            
            return 1

        _log.info(f'###### This is step {overall_epoch_i} of {max_iters*len(trainloader)} steps ######')
        for _, sample_batched in enumerate(trainloader):
            
            ## To verify which labels and PIDs are used for training
            qry_pid = sample_batched['query_scan_z_ids'][0][0][0]
            supp_pid = sample_batched['supp_scan_z_ids'][0][0][0]
            curr_lbl = sample_batched['class_ids'][0].cpu().numpy()[0]

            pids_used.add(qry_pid)
            pids_used.add(supp_pid)
            labels_train.add(curr_lbl)
            supp_cls_pid_idx[curr_lbl].add(supp_pid)
            
            # Prepare input
            overall_epoch_i += 1
            ## Inputs For FSS 
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
            ## Inputs for contrastive branch
            ## Create 2 augmented views of the query image

            # Concatenate to get augmentations, currently only takes all labels in 1 channel : Change later to support mutiple channels for each label
            # For augmentation input
            query_img = query_images[0] # (1,3,256,256)
            query_img = query_img[0,0,:,:].unsqueeze(-1).cpu().numpy() # (256,256,1)

            if len(sample_batched['query_all_lbls'].squeeze().shape) > 2: # OH labels
                query_all_lbls = sample_batched['query_all_lbls'].squeeze().cpu().numpy() #(256,256,5)

            else: 
                query_all_lbls = sample_batched['query_all_lbls'].squeeze().unsqueeze(-1).cpu().numpy() # (256,256,1)
            
            comp = np.concatenate( [query_img, query_all_lbls], axis = -1 ) # (256,256,1) both 
            
            ## Returns 1-hot encoded labels (256,256,num_classes)
            aug_img_1, aug_lbl_oh_1 = tr_transforms(comp, c_img = 1, c_label = 1, nclass = max_label + 1,  is_train = True, use_onehot = True)
            aug_img_2, aug_lbl_oh_2 = tr_transforms(comp, c_img = 1, c_label = 1, nclass = max_label + 1,  is_train = True, use_onehot = True)
            
            aug_img_1 = torch.from_numpy( np.transpose( aug_img_1, (2, 0, 1)) )
            aug_img_1 = aug_img_1.repeat( [ 3, 1, 1] )
            aug_img_1 = aug_img_1.unsqueeze(0)

            aug_img_2 = torch.from_numpy( np.transpose( aug_img_2, (2, 0, 1)) )
            aug_img_2 = aug_img_2.repeat( [ 3, 1, 1] )
            aug_img_2 = aug_img_2.unsqueeze(0)

            aug_lbl_oh_1 = np.expand_dims(aug_lbl_oh_1,axis=0)
            aug_lbl_oh_2 = np.expand_dims(aug_lbl_oh_2,axis=0)

            # Need numpy arrays for index calc
            img_cat_batch = np.concatenate([aug_img_1.numpy(), aug_img_2.numpy()], axis=0)
            lbl_cat_batch = np.concatenate([aug_lbl_oh_1,aug_lbl_oh_2], axis=0)
            

            # Returns pos and neg indices as follows :
            # [ for_each image, [ for_each class, [ for_each positive/negative element, [X & Y coord of the pixel] ] ] ]
            # Whole array will be 1000 if the class in not present in current image : Skip that class while calc. contrastive loss
            net_pos_ele1_arr,net_pos_ele2_arr,net_neg_ele1_arr,net_neg_ele2_arr = calc_pos_neg_index(lbl_cat_batch,batch_size_ft=1,num_classes=max_label+1,no_of_pos_eles=3,no_of_neg_eles=3)

            # Convert to torch for network
            img_cat_batch = torch.from_numpy(img_cat_batch).cuda()
            lbl_cat_batch = torch.from_numpy(lbl_cat_batch).cuda()

            # Concat all positive indices 
            pos_arr= np.concatenate((net_pos_ele1_arr,net_pos_ele2_arr),axis=0) # (2,class,no_pos_ele,2)
            # Concat all negative indices
            neg_arr= np.concatenate((net_neg_ele1_arr,net_neg_ele2_arr),axis=0) # (2,`class,no_neg_ele,2)


            ## Use +ve and -ve indices calc above to get the contrastive loss
            # calculate pos and neg indices of masks
            pos_count_arr1 = np.count_nonzero(np.where(net_pos_ele1_arr != 1000))
            pos_count_arr2 = np.count_nonzero(np.where(net_pos_ele2_arr != 1000))

            neg_count_arr1 = np.count_nonzero(np.where(net_neg_ele1_arr != 1000))
            neg_count_arr2 = np.count_nonzero(np.where(net_neg_ele2_arr != 1000))

            if (pos_count_arr1 == 0 or pos_count_arr2 == 0 or neg_count_arr1 == 0 or neg_count_arr2 == 0):
                continue
            
            ## Get segmentation loss and contrastive loss
            try:
                query_pred, align_loss, debug_vis, assign_mats = model_seg(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            except:
                print("Faulty batch detected, skip!")
                continue


            # query_pred, align_loss, debug_vis, assign_mats = model_seg(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
            

            y_fin_out = model_cont(img_cat_batch)
            try:
                cont_loss,num_i1_ss,num_i2_ss,den_i1_ss,den_i2_ss = local_cont_loss(y_fin=y_fin_out,y_l_reg=lbl_cat_batch,pos_indx=pos_arr,neg_indx=neg_arr,dataset=baseset_name)   

                if (math.isnan(cont_loss) == True or cont_loss == 0):
                    # print('continue,epoch_i', overall_epoch_i)
                    continue

                if (num_i1_ss == 0 or num_i2_ss == 0 or den_i1_ss == 0 or den_i2_ss == 0):
                    # print('continue,epoch_i', overall_epoch_i)
                    continue
            except:
                print('Fault in contrastive loss calculation, skip')
                continue
            
            query_loss = criterion(query_pred, query_labels)
            seg_loss = query_loss + align_loss
            net_cont_loss = lambda_local*cont_loss
            
            total_loss = seg_loss + net_cont_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # Log loss
            seg_loss = seg_loss.detach().data.cpu().numpy()
            cont_loss = net_cont_loss.detach().data.cpu().numpy()
            total_loss = total_loss.detach().data.cpu().numpy()[0]

            _run.log_scalar('total_loss', total_loss)
            _run.log_scalar('seg_loss', seg_loss)
            _run.log_scalar('cont_loss', cont_loss)

            log_loss['total_loss'] += total_loss
            log_loss['seg_loss'] += seg_loss
            log_loss['cont_loss'] += cont_loss
            # print loss and take snapshots
            if (overall_epoch_i + 1) % _config['print_interval'] == 0:

                total_loss = log_loss['total_loss'] / _config['print_interval']
                seg_loss = log_loss['seg_loss'] / _config['print_interval']
                cont_loss = log_loss['cont_loss'] / _config['print_interval']
                writer.add_scalar('train/total_loss', total_loss, overall_epoch_i + 1)
                writer.add_scalar('train/seg_loss', seg_loss, overall_epoch_i + 1)
                writer.add_scalar('train/cont_loss', cont_loss, overall_epoch_i + 1)

                log_loss['total_loss'] = 0
                log_loss['seg_loss'] = 0
                log_loss['cont_loss'] = 0

                print(f'step {overall_epoch_i+1}: total_loss: {total_loss}, seg_loss: {seg_loss}, cont_loss: {cont_loss}')

            # Save model based on validation dice score
            if (overall_epoch_i + 1) % _config['save_snapshot_every'] == 0:
                # Get dice score on validation set
                curr_val_dsc = get_unlbl_pseudo_lbls_sep(model=model_seg,sup_dataset = tr_parent,norm_func = norm_func,unlbl_ids=val_pids,_config=_config,_log=_log,val_mode=True)  
                model_seg.train()

                if curr_val_dsc > max_val_dsc:
                    max_val_dsc = curr_val_dsc
                    _log.info('###### Taking snapshot ######')
                    torch.save(model_seg.state_dict(),
                            os.path.join(f'{_run.observers[0].dir}/snapshots', f'seg_best.pth'))
                
                writer.add_scalar('train/best_val_dsc', max_val_dsc, overall_epoch_i + 1)
                writer.add_scalar('train/val_dsc', curr_val_dsc, overall_epoch_i + 1)