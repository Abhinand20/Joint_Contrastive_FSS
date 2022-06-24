"""
Utils for datasets
"""
import numpy as np

import os
import sys
import nibabel as nib
import numpy as np
import pdb
import SimpleITK as sitk
import glob
import json
import re
from util.utils import CircularList

sys.path.insert(0, '../../dataloaders/')

DATASET_INFO = {
    "CHAOST2": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],
            'REAL_LABEL_NAME': ["BG", "LIVER", "RK", "LK", "SPLEEN"],
            '_SEP': [0, 4, 8, 12, 16, 20], # 4 groups of 4, 4, 4, 4 : 4-fold CV (If using 4 labeled volumes)
            '_SEP1' : [0, 4, 9, 14, 19], # 4 groups of 4, 5, 5, 5 : 4-fold CV (If using 1 labeled volume)
            '_SEP2' : [0, 3, 8, 13, 18], # 4 groups of 3, 5, 5, 5 : 4-fold CV (If using 2 labeled volumes)
            'MODALITY': 'MR',
            'LABEL_GROUP': {
                'pa_all': set(range(1, 5)),
                0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
                1: set([2, 3]), # lower_abdomen
                },
            # 'LBL_PIDS' : ['1','2','3','5']
            'LBL_PIDS' : ['5']
            },

    "SABS": {
            'PSEU_LABEL_NAME': ["BGD", "SUPFG"],

            'REAL_LABEL_NAME': ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC",\
              "PS_VEIN", "PANCREAS", "AG_R", "AG_L"],
            # '_SEP': [0, 6, 12, 18, 24, 30], 
            '_SEP1': [0, 6, 12, 18, 24, 29], # 29 images: 6,6,6,6,5 : 5-fold CV 
            'MODALITY': 'CT',
            'LABEL_GROUP':{
                'pa_all': set( [1,2,3,6]  ),
                0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
                1: set( [2,3] ), # lower_abdomen
                    },
            'LBL_PIDS' : ['3']
            }

}

def read_nii_bysitk(input_fid, peel_info = False):
    """ read nii to numpy through simpleitk

        peelinfo: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if peel_info:
        info_obj = {
                "spacing": img_obj.GetSpacing(),
                "origin": img_obj.GetOrigin(),
                "direction": img_obj.GetDirection(),
                "array_size": img_np.shape
                }
        return img_np, info_obj
    else:
        return img_np

def get_normalize_op(modality, fids):
    """
    As title
    Args:
        modality:   CT or MR
        fids:       fids for the fold
    """

    def get_CT_statistics(scan_fids):
        """
        As CT are quantitative, get mean and std for CT images for image normalizing
        As in reality we might not be able to load all images at a time, we would better detach statistics calculation with actual data loading
        """
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix

        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2 )
            del in_img
        var_all = total_var / n_pix

        global_std = var_all ** 0.5

        return meanval, global_std

    if modality == 'MR':

        def MR_normalize(x_in):
            return (x_in - x_in.mean()) / x_in.std()

        return MR_normalize #, {'mean': None, 'std': None} # we do not really need the global statistics for MR

    elif modality == 'CT':
        ct_mean, ct_std = get_CT_statistics(fids)
        # debug
        print(f'###### DEBUG_DATASET CT_STATS NORMALIZED MEAN {ct_mean / 255} STD {ct_std / 255} ######')

        def CT_normalize(x_in):
            """
            Normalizing CT images, based on global statistics
            """
            return (x_in - ct_mean) / ct_std

        return CT_normalize #, {'mean': ct_mean, 'std': ct_std}

def read_nii_bysitk(input_fid, peel_info = False):
    """ read nii to numpy through simpleitk
        peelinfo: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if len(img_np.shape) == 3:
        if peel_info:
            info_obj = {
                    "spacing": img_obj.GetSpacing(),
                    "origin": img_obj.GetOrigin(),
                    "direction": img_obj.GetDirection(),
                    "array_size": img_np.shape
                    }
            return img_np, info_obj
        else:
            return img_np
    
    else:
        return img_np

def update_class_slice_index(dataset):
    MIN_TP = 100 # minimum number of positive label pixels to be recorded
    if dataset == 'SABS':
        IMG_BNAME="./data/SABS/sabs_CT_normalized/image_*.nii.gz"
        SEG_BNAME="./data/SABS/sabs_CT_normalized/label_*.nii.gz"
        LABEL_NAME = ["BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS", "LIVER", "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS", "AG_R", "AG_L"]   
        fid = f'./data/SABS/sabs_CT_normalized/classmap_{MIN_TP}.json' # name of the output file. 


    elif dataset == 'CHAOST2':
        IMG_BNAME="./data/CHAOST2/chaos_MR_T2_normalized/image_*.nii.gz"
        SEG_BNAME="./data/CHAOST2/chaos_MR_T2_normalized/label_*.nii.gz"
        LABEL_NAME = ["BG", "LIVER", "RK", "LK", "SPLEEN"]     
        fid = f'./data/CHAOST2/chaos_MR_T2_normalized/classmap_{MIN_TP}.json' # name of the output file. 

    imgs = glob.glob(IMG_BNAME)
    segs = glob.glob(SEG_BNAME)
    imgs = [ fid for fid in sorted(imgs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]
    segs = [ fid for fid in sorted(segs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]
    classmap = {}

    for _lb in LABEL_NAME:
        classmap[_lb] = {}
        for _sid in segs:
            pid = _sid.split("_")[-1].split(".nii.gz")[0]
            classmap[_lb][pid] = []

    for seg in segs:
        pid = seg.split("_")[-1].split(".nii.gz")[0]
        lb_vol = read_nii_bysitk(seg)
        # When we have compact labels
        if len(lb_vol.shape) == 3:
            n_slice = lb_vol.shape[0]
            for slc in range(n_slice):
                for cls in range(len(LABEL_NAME)):
                    if cls in lb_vol[slc, ...]:
                        if np.sum( lb_vol[slc, ...]) >= MIN_TP:
                            classmap[LABEL_NAME[cls]][str(pid)].append(slc)
        
        # When we have labels in separate channels
        elif len(lb_vol.shape) == 4:
            n_slice = lb_vol.shape[0]
            for slc in range(n_slice):
                for cls in range(len(LABEL_NAME)):
                    curr_slice = lb_vol[slc,cls,...]
                    if np.sum( curr_slice) >= MIN_TP:
                        classmap[LABEL_NAME[cls]][str(pid)].append(slc)
        
    with open(fid, 'w') as fopen:
        json.dump(classmap, fopen)
        fopen.close()  
    
    print("Class slice index updated!")
<<<<<<< HEAD

def organize_sample_fids(base_dir,scan_ids):
    out_list = {}
    for curr_id in scan_ids:
        curr_dict = {}

        _img_fid = os.path.join(base_dir, f'image_{curr_id}.nii.gz')
        _lb_fid  = os.path.join(base_dir, f'label_{curr_id}.nii.gz')

        curr_dict["img_fid"] = _img_fid
        curr_dict["lbs_fid"] = _lb_fid
        out_list[str(curr_id)] = curr_dict

    return out_list


def get_CT_norm_op(which_dataset, base_dir, tr_pids):
    img_modality = DATASET_INFO[which_dataset]['MODALITY']
    sep = DATASET_INFO[which_dataset]['_SEP1']
    pseu_label_name = DATASET_INFO[which_dataset]['PSEU_LABEL_NAME']
    real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

    nclass = len(pseu_label_name)

    # find scans in the data folder
    img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(base_dir + "/image_*.nii.gz") ]
    img_pids = CircularList(sorted( img_pids, key = lambda x: int(x)))

    # experiment configs
    # img_pids are just the ones that are passed from outside (Patient IDs of Labeled and Unlabeled sets)
    if tr_pids == None: # consider all images
        img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(base_dir + "/image_*.nii.gz") ]
    else: 
        img_pids = tr_pids

    img_lb_fids = organize_sample_fids(base_dir,img_pids) # information of scans of the entire fold
    norm_func = get_normalize_op(img_modality, [ fid_pair['img_fid'] for _, fid_pair in img_lb_fids.items()])

    return norm_func
=======
>>>>>>> fcdf4fa4fa582f43f48fe1b4116ac09a50ee1479
