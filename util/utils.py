"""Util functions
Extended from original PANet code
TODO: move part of dataset configurations to data_utils
"""
import random
import torch
import numpy as np
import operator
import numpy as np
import os
import glob
import SimpleITK as sitk
import sys
import json
sys.path.insert(0, '../../dataloaders/')

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'SABS': {
        'pa_all': set( [1,2,3,6]  ),
        0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
        1: set( [2,3] ), # lower_abdomen
    },
    'C0': {
        'pa_all': set(range(1, 4)),
        0: set([2,3]),
        1: set([1,3]),
        2: set([1,2]),
    },
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]), # lower_abdomen
    },
}

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()

def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)

def compose_wt_simple(is_wce, data_name):
    """
    Weights for cross-entropy loss
    """
    if is_wce:
        if data_name in ['SABS', 'SABS_Superpix', 'C0', 'C0_Superpix', 'CHAOST2', 'CHAOST2_Superpix']:
            return torch.FloatTensor([0.05, 1.0]).cuda()
        else:
            raise NotImplementedError
    else:
        return torch.FloatTensor([1.0, 1.0]).cuda()


class CircularList(list):
    """
    Helper for spliting training and validation scans
    Originally: https://stackoverflow.com/questions/8951020/pythonic-circular-list/8951224
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)

def calc_pos_neg_index(ip_mask, batch_size_ft, num_classes, no_of_pos_eles, no_of_neg_eles):
    """
    ip_mask is of the shape : (batch_size_ft,H,W,num_classes) -> 1 mask for each class

    Returns the +ve and -ve indices in the images for each augmented pair of images in a batch

    [ for_each image, [ for_each class, [ for_each positive/negative element, [X & Y coord of the pixel] ] ] ]

    -ve indices -> total elements = no_of_neg_eles * (num_classes - 2) : we are omitting the current class and including all others
    """
    net_pos_ele1_arr = np.zeros((batch_size_ft, num_classes - 1, no_of_pos_eles, 2))
    net_pos_ele2_arr = np.zeros((batch_size_ft, num_classes - 1, no_of_pos_eles, 2))
    net_neg_ele1_arr = np.zeros((batch_size_ft, num_classes - 1, no_of_neg_eles * (num_classes - 2), 2))
    net_neg_ele2_arr = np.zeros((batch_size_ft, num_classes - 1, no_of_neg_eles * (num_classes - 2), 2))

    net_pos_ele1_arr[:], net_pos_ele2_arr[:], net_neg_ele1_arr[:], net_neg_ele2_arr[:] = 1000, 1000, 1000, 1000

    ## 2 augmentation of the same image batch_size_ft indices apart
    for pos_index in range(0, batch_size_ft, 1):
        
        ## 2 lists to store the masks of augmentations of the same image
        i1_ele_list, i2_ele_list = [], []
        ## 1st augmentation
        index_pos1 = pos_index
        ## 2nd augmentation
        index_pos2 = batch_size_ft + pos_index
        ## For each image, loop over all the classes, get the mask and treat it as the positive pair
        for cls_index in range(0, num_classes):
            
            ## Mask to get the features of the curr_class (np.where returns Indices of where the mask != 0)
            ## Tuple of indices where condition is true ( [x_coord] , [y_coord] )
            indices = np.where(ip_mask[index_pos1, :, :, cls_index] > 0)
            
            ## Add the mask of 1st image
            i1_ele_list.append(indices)


            ## Same as above for 2nd augmentation
            indices = np.where(ip_mask[index_pos2, :, :, cls_index] > 0)
            
            i2_ele_list.append(indices)
        
        ## TILL HERE : We have the masks stored for each class for each image in i1_ele_list,i2_ele_list
        ## Each image is separated by num_classes index [<im1_c1>,<im1_c2>,<im2_c1>,<im2_c2>] (num_classes = 2 here)


        ## Positive elements are just the corresponding masks at the index i --> +ve : [<i1_ele_list[i]>,<i2_ele_list[i]>]
        ## We actually need to pick just 3 pixels from each +ve mask and use it
        for cls_index_pos in range(1, num_classes):
            # pos_ele1_list,pos_ele2_list = [],[]
            # print('cls_index',cls_index_pos)
            
            ## Store the current positive mask
            pos_ele1_list = i1_ele_list[cls_index_pos]
            pos_ele2_list = i2_ele_list[cls_index_pos]
            # print('test',np.shape(pos_ele1_list),len(pos_ele1_list),np.shape(pos_ele1_list)[1])

            #     print('pos_ele1_list',pos_ele1_list,np.shape(pos_ele1_list)[0])
            #     print('pos_ele2_list',pos_ele2_list,np.shape(pos_ele2_list)[0])
            
            ## If the current image slice does not contain a curr_class, we should skip it
            if (np.shape(pos_ele1_list)[1] == 0 or np.shape(pos_ele2_list)[1] == 0):
                # print('0 elements in pos mask')
                continue
            
            ## Get the negative masks : Reset for each new image
            all_neg_ele1_list, all_neg_ele2_list = [], []
            
            ## Negative will be all classes other than current class
            for cls_index_neg in range(1, num_classes):
                if (cls_index_neg != cls_index_pos):
                    # print('-',cls_index_pos,cls_index_neg)
                    # if(np.shape(pos_ele1_list)[1]==0 or np.shape(pos_ele2_list)[1]==0):
                    #    print('0 elements in neg mask')
                    #    continue

                    ## Store the negative classes for both augmentations
                    all_neg_ele1_list.append(i1_ele_list[cls_index_neg])
                    all_neg_ele2_list.append(i2_ele_list[cls_index_neg])

            ## Pick 3 random indices from the available +ve indices (These are the 3 pixels that we will consider)
            pos_ele1 = np.random.randint(0, high=np.shape(pos_ele1_list)[1], size=no_of_pos_eles, dtype=np.int32)
            pos_ele2 = np.random.randint(0, high=np.shape(pos_ele2_list)[1], size=no_of_pos_eles, dtype=np.int32)
            

            ## TILL HERE : We have the 3 positive pixels for both elements and the negative masks stored

            # net_pos_ele1_list.append([0,cls_index_pos])
            # pos1_tmp_list,pos2_tmp_list=[],[]
            
            ## Processing each pixel now
            for p1 in range(no_of_pos_eles):
                pos_index1 = pos_ele1[p1]
                pos_index2 = pos_ele2[p1]
                # print(pos_ele1[p1],pos_ele2[p1])
                # print('+ve index, value ',pos_index1,' - ',pos_ele1_list[0][pos_index1],pos_ele1_list[1][pos_index1])

                # print('+ve value',
                # net_pos_ele1_list.append([0,cls_index_pos,[pos_ele1_list[0][pos_index1],pos_ele1_list[1][pos_index1]]])
                # pos1_tmp_list.append([pos_ele1_list[0][pos_index1],pos_ele1_list[1][pos_index1]])
                # pos2_tmp_list.append([pos_ele2_list[0][pos_index2],pos_ele2_list[1][pos_index2]])

                # print('++',index_pos1,cls_index_pos-1,p1,0)
                
                ## Populate the +ve pixel values for the 2 positive pairs (2 Augmentations of the same image)
                
                ## 1st +ve elem
                ## Populate row_index
                net_pos_ele1_arr[index_pos1][cls_index_pos - 1][p1][0] = pos_ele1_list[0][pos_index1]
                ## Populate col_index
                net_pos_ele1_arr[index_pos1][cls_index_pos - 1][p1][1] = pos_ele1_list[1][pos_index1]

                ## 2nd +ve elem
                net_pos_ele2_arr[index_pos1][cls_index_pos - 1][p1][0] = pos_ele2_list[0][pos_index2]
                net_pos_ele2_arr[index_pos1][cls_index_pos - 1][p1][1] = pos_ele2_list[1][pos_index2]


            ## Calculate the -ve indices now
            
            ## all_neg_ele1_list : Contains the indices all -ve elements of curr_img (len = num_classes-1)

            ## LOGIC : Same as above, loop over all -ve's of the current class, pick 3 pixels and store them
            n1_c = 0
            ## Loop over all -ve elements
            for cls_index_neg in range(0, len(all_neg_ele1_list)):
                ## Store curr_negative
                neg_ele1_list = all_neg_ele1_list[cls_index_neg]
                # print('neg shape',cls_index_pos,cls_index_neg,len(all_neg_ele1_list),np.shape(neg_ele1_list)[1])
                # print('neg values',neg_ele1_list)
                if (np.shape(neg_ele1_list)[1] == 0):
                    # print('0 elements in neg1 mask')
                    n1_c = n1_c + (no_of_neg_eles)
                    continue
                
                ## Pick 3 random pixels/indices
                neg_ele1 = np.random.randint(0, high=np.shape(neg_ele1_list)[1], size=no_of_neg_eles, dtype=np.int32)
                # neg_ele1 = np.random.randint(0, high=np.shape(neg_ele1_list)[1], size=no_of_neg_eles*(num_classes-1), dtype=np.int32)
                # print('neg_ele1',neg_ele1)

                # neg_tmp_list=[]

                ## Iterate over the 3 pixels and store the values
                for n1 in range(no_of_neg_eles):
                    # for n1 in range(no_of_neg_eles*(num_classes-1)):
                    neg_index1 = neg_ele1[n1]
                    # print(pos_ele1[p1],pos_ele2[p1])
                    # print('-ve index, value',neg_index1,' - ',neg_ele1_list[0][neg_index1],neg_ele1_list[1][neg_index1])
                    # print('-ve value',n1_c,neg_ele1_list[0][neg_index1],neg_ele1_list[1][neg_index1])
                    # net_neg_ele1_list.append([0,[cls_index_neg],[neg_ele1_list[0][neg_index1],neg_ele1_list[1][neg_index1]]])
                    # neg_tmp_list.append([neg_ele1_list[0][neg_index1],neg_ele1_list[1][neg_index1]])

                    # print('--',index_pos1,cls_index_neg,n1_c,0,n1)

                    net_neg_ele1_arr[index_pos1][cls_index_pos - 1][n1_c][0] = neg_ele1_list[0][neg_index1]
                    net_neg_ele1_arr[index_pos1][cls_index_pos - 1][n1_c][1] = neg_ele1_list[1][neg_index1]
                    n1_c = n1_c + 1
                    # print('net_neg_ele1_arr',net_neg_ele1_arr)
                # net_neg_ele1_list.append([0,cls_index_neg,tmp_list])
                # net_pos_ele1_list.append([neg_tmp_list])
                # net_neg_ele1_list.append(neg_tmp_list)
            # net_pos_ele1_list.append([0,cls_index_pos,pos1_tmp_list,net_neg_ele1_list])

            n2_c = 0
            for cls_index_neg in range(0, len(all_neg_ele2_list)):
                neg_ele2_list = all_neg_ele2_list[cls_index_neg]
                if (np.shape(neg_ele2_list)[1] == 0):
                    # print('0 elements in neg2 mask')
                    n2_c = n2_c + (no_of_neg_eles)
                    continue

                neg_ele2 = np.random.randint(0, high=np.shape(neg_ele2_list)[1], size=no_of_neg_eles, dtype=np.int32)
                # print('neg_ele2',neg_ele2)

                # neg_tmp_list=[]
                for n2 in range(no_of_neg_eles):
                    neg_index2 = neg_ele2[n2]
                    # print(pos_ele1[p1],pos_ele2[p1])
                    # print('-ve index',neg_index2)
                    # print('-ve value',neg_ele2_list[0][neg_index2],neg_ele2_list[1][neg_index2])
                    # net_neg_ele2_list.append(neg_tmp_list)
                    # print('--',index_pos1,cls_index_neg,n2_c,0,n2)

                    #net_neg_ele2_arr[index_pos1][cls_index_neg][n2_c][0] = neg_ele2_list[0][neg_index2]
                    #net_neg_ele2_arr[index_pos1][cls_index_neg][n2_c][1] = neg_ele2_list[1][neg_index2]
                    net_neg_ele2_arr[index_pos1][cls_index_pos - 1][n2_c][0] = neg_ele2_list[0][neg_index2]
                    net_neg_ele2_arr[index_pos1][cls_index_pos - 1][n2_c][1] = neg_ele2_list[1][neg_index2]
                    n2_c = n2_c + 1

            # net_pos_ele2_list.append([0,cls_index_pos,pos2_tmp_list,net_neg_ele2_list])

    # net_pos_arr= np.concatenate((net_pos_ele1_arr,net_pos_ele2_arr),axis=0)
    # net_neg_arr= np.concatenate((net_neg_ele1_arr,net_neg_ele2_arr),axis=0)

    return net_pos_ele1_arr, net_pos_ele2_arr, net_neg_ele1_arr, net_neg_ele2_arr
    # return net_pos_arr,net_neg_arr

