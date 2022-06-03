from operator import index
import torch
import numpy as np
import torch.nn.functional as F

def cos_sim(vec_a,vec_b,temp_factor = 0.1):
    

    cos_sim_ans = F.cosine_similarity(vec_a,vec_b,dim=-1) / temp_factor
    return cos_sim_ans

def local_cont_loss(y_fin,y_l_reg,pos_indx,neg_indx,num_filters=16,batch_size=1,local_loss_exp_no=0):
    """
    Calculate the contrastive loss (Intra only for now)
    Inputs:
        y_fin : Output of contrastive branch (2,16,H,W)
        pos_idx : Positive indices for selected pixels (2,num_class,num_pos_elem,2)
        y_l_reg : Labels for the image (H,W,num_classes) - check this; pos/neg indices have num_class - 1 (no background)
        neg_idx : Negative indices for selected pixels 
        batch_size : 1 for now
        local_loss_exp_no : 0 -> Intra ; 1 -> Inter
    """

    local_loss=0
    net_local_loss=0
    num_pos_elem = pos_indx.shape[-2]
    num_classes = pos_indx.shape[-3]
    ## Iterate over the images
    for pos_index in range(batch_size):
        index_pos1 = pos_index
        index_pos2 = pos_index + batch_size

        # Get the embeddings of both images
        x_num_i1 = y_fin[index_pos1]
        x_num_i2 = y_fin[index_pos2]

        ## Zero vector for cases when labels don't exist
        x_zero_vec = torch.zeros(1,num_filters)
        
        mask_i1 = y_l_reg[index_pos1]
        mask_i2 = y_l_reg[index_pos2]

        ## Hardcoded right now for MRI dataset
        # Index for positive classes (4 classes: LK,RK,Spleen,Liver)

        ## To get +ve class use pos_indx[pos_class]
        ## To get mask of that class : mask_i1[pos_class+1] (Bcs mask has background too)
        ## For negative
        pos_cls_ref = np.asarray([0,1,2,3])
        neg_cls_ref = np.asarray([1,2,3,4])

        for pos_cls in pos_cls_ref:
            
            # Store the pixel coords of pos_cls
            pos_cls_ele_i1 = pos_indx[index_pos1][pos_cls]
            pos_cls_ele_i2 = pos_indx[index_pos2][pos_cls]

            # Get the mask of positive class
            mask_i1_pos = mask_i1[...,pos_cls+1]
            mask_i1_pos = mask_i1_pos.repeat(num_filters,1,1)
            # Calculate the average of pos embeddings
            # Apply mask over all filters and reshape to (num_filters,-1)
            pos_cls_avg_i1 = x_num_i1[mask_i1_pos.bool()].reshape(num_filters,-1)
            # Take average of embeddings for each filter dimension
            pos_avg_vec_i1_p = torch.mean(pos_cls_avg_i1,axis=1).reshape(1,num_filters)

            ## Same logic as above - Masking the negative classes , getting the mean embeddings and storing into a list
            #############################
            # make list of negative classes masks' mean embeddings from image 1 (x) mask
            neg_mask1_list = []
            for neg_cls_i1 in neg_cls_ref:
                mask_i1_neg = mask_i1[...,neg_cls_i1]
                mask_i1_neg = mask_i1_neg.repeat(num_filters,1,1)
                neg_cls_avg_i1 =  x_num_i1[mask_i1_neg.bool()].reshape(num_filters,-1)
                neg_avg_vec_i1_p = torch.mean(neg_cls_avg_i1,axis=1).reshape(1,num_filters)
                neg_mask1_list.append(neg_avg_vec_i1_p)
            #print('neg_mask1_list', neg_mask1_list)
            #############################


            ## Do the same for image 2
            #############################
            #mask of image 2 (x')  from batch X_B
            #select positive classes masks' mean embeddings

            mask_i2_pos = mask_i2[...,pos_cls+1]
            mask_i2_pos = mask_i2_pos.repeat(num_filters,1,1)
            # Calculate the average of pos embeddings
            # Apply mask over all filters and reshape to (num_filters,-1)
            pos_cls_avg_i2 = x_num_i2[mask_i2_pos.bool()].reshape(num_filters,-1)
            # Take average of embeddings for each filter dimension
            pos_avg_vec_i2_p = torch.mean(pos_cls_avg_i2,axis=1).reshape(1,num_filters)

            ## Same logic as above - Masking the negative classes , getting the mean embeddings and storing into a list
            #############################
            # make list of negative classes masks' mean embeddings from image 1 (x) mask
            neg_mask2_list = []
            for neg_cls_i2 in neg_cls_ref:
                mask_i2_neg = mask_i2[...,neg_cls_i2]
                mask_i2_neg = mask_i2_neg.repeat(num_filters,1,1)
                neg_cls_avg_i2 =  x_num_i2[mask_i2_neg.bool()].reshape(num_filters,-1)
                neg_avg_vec_i2_p = torch.mean(neg_cls_avg_i2,axis=1).reshape(1,num_filters)
                neg_mask2_list.append(neg_avg_vec_i2_p)
            #print('neg_mask1_list', neg_mask1_list)
            #############################


            ## Now compare the individual pixel embeddings with the entire mean of the classes
            # Loop over all selected positive pixels
            for n_pos_idx in range(pos_indx.shape[-2]):
                x_coord_i1 = int(pos_cls_ele_i1[n_pos_idx][0])
                y_coord_i1 = int(pos_cls_ele_i1[n_pos_idx][1])

                ## This slice doesn't contain the current class : skip
                if (x_coord_i1 == 1000 or y_coord_i1==1000):
                    continue

                x_num_tmp_i1 = x_num_i1[:,x_coord_i1,y_coord_i1].reshape(1,num_filters)

                x_n1_count = len(x_num_tmp_i1[x_num_tmp_i1!=0])
                ## Handle the case when all features are 0
                if x_n1_count > 0:
                    x_w3_n_i1 = x_num_tmp_i1
                else:
                    x_w3_n_i1 = x_zero_vec


                ## Do the same for image 2
                x_coord_i2 = int(pos_cls_ele_i2[n_pos_idx][0])
                y_coord_i2 = int(pos_cls_ele_i2[n_pos_idx][1])

                if (x_coord_i2 == 1000 or x_coord_i2==1000):
                    continue
                
                x_num_tmp_i2 = x_num_i2[:,x_coord_i2,y_coord_i2].reshape(1,num_filters)

                x_n2_count = len(x_num_tmp_i2[x_num_tmp_i2!=0])
                ## Handle the case when all features are 0
                if x_n2_count > 0:
                    x_w3_n_i2 = x_num_tmp_i2
                else:
                    x_w3_n_i2 = x_zero_vec

                # NUMERATOR terms - cosine similarity between +ve pair
                pos_avg_vec_i1=pos_avg_vec_i1_p
                pos_avg_vec_i2=pos_avg_vec_i2_p 

                ## Image 1
                log_or_n1 = (torch.equal(torch.count_nonzero(x_num_tmp_i1),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(pos_avg_vec_i1),torch.tensor(0).cuda()))
                log_or_n1_nan = bool(torch.logical_or(torch.isnan(torch.sum(x_w3_n_i1)),torch.isnan(torch.sum(pos_avg_vec_i1))))

                log_or_n1_net = (log_or_n1 or log_or_n1_nan)
                if log_or_n1_net:
                    num_i1_ss = torch.tensor(0.0).cuda()
                else:
                    num_i1_ss = cos_sim(x_w3_n_i1,pos_avg_vec_i1)        
                
                # Image 2
                log_or_n2 = (torch.equal(torch.count_nonzero(x_num_tmp_i2),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(pos_avg_vec_i2),torch.tensor(0).cuda()))
                log_or_n2_nan = bool(torch.logical_or(torch.isnan(torch.sum(x_w3_n_i2)),torch.isnan(torch.sum(pos_avg_vec_i2))))

                log_or_n2_net = (log_or_n2 or log_or_n2_nan)
                if log_or_n2_net:
                    num_i2_ss = torch.tensor(0.0).cuda()
                else:
                    num_i2_ss = cos_sim(x_w3_n_i2,pos_avg_vec_i2) 

                ## DENOMINATOR terms!!  
                den_i1_ss,den_i2_ss=0,0
                # Image 1
                for neg_avg_i1_c1 in neg_mask1_list:
                    log_or_n1_d1 = (torch.equal(torch.count_nonzero(x_num_tmp_i1),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(neg_avg_i1_c1),torch.tensor(0).cuda()))
                    log_or_n1_d1_nan = bool(torch.logical_or(torch.isnan(torch.sum(x_w3_n_i1)),torch.isnan(torch.sum(neg_avg_i1_c1))))

                    log_or_n1_d1_net = (log_or_n1_d1 or log_or_n1_d1_nan)
                    if log_or_n1_d1_net:
                        den_i1_ss = den_i1_ss + torch.tensor(0.0).cuda()
                    else:
                        tmp = torch.exp(cos_sim(x_w3_n_i1,neg_avg_i1_c1)) 
                        den_i1_ss = den_i1_ss + tmp
                
                # Image 2
                for neg_avg_i2_c2 in neg_mask2_list:
                    log_or_n2_d2 = (torch.equal(torch.count_nonzero(x_num_tmp_i2),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(neg_avg_i2_c2),torch.tensor(0).cuda()))
                    log_or_n2_d2_nan = bool(torch.logical_or(torch.isnan(torch.sum(x_w3_n_i2)),torch.isnan(torch.sum(neg_avg_i2_c2))))

                    log_or_n2_d2_net = (log_or_n2_d2 or log_or_n2_d2_nan)
                    if log_or_n2_d2_net:
                        den_i2_ss = den_i2_ss + torch.tensor(0.0).cuda()
                    else:
                        den_i2_ss = den_i2_ss + torch.exp(cos_sim(x_w3_n_i2,neg_avg_i2_c2))

                ## Calculate the final loss 
                ## Image 1
                log_num_i1_nan = bool(torch.logical_or(torch.isnan(torch.exp(num_i1_ss)),torch.isnan(torch.exp(den_i1_ss))))

                log_num_i1_zero = (torch.equal(torch.count_nonzero(num_i1_ss),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(den_i1_ss),torch.tensor(0).cuda()))
                log_num_i1_net = (log_num_i1_nan or log_num_i1_zero)

                if log_num_i1_net:
                    num_i1_loss = torch.tensor(0.0).cuda()
                else:
                    num_i1_loss = -torch.log( torch.exp(num_i1_ss) / den_i1_ss)
                
                local_loss = local_loss + num_i1_loss

                ## Image 2
                log_num_i2_nan = bool(torch.logical_or(torch.isnan(torch.exp(num_i2_ss)),torch.isnan(torch.exp(den_i2_ss))))

                log_num_i2_zero = (torch.equal(torch.count_nonzero(num_i2_ss),torch.tensor(0).cuda()) or torch.equal(torch.count_nonzero(den_i2_ss),torch.tensor(0).cuda()))
                log_num_i2_net = (log_num_i2_nan or log_num_i2_zero)

                if log_num_i2_net:
                    num_i2_loss = torch.tensor(0.0).cuda()
                else:
                    num_i2_loss = -torch.log( torch.exp(num_i2_ss) / den_i2_ss)
                
                local_loss = local_loss + num_i2_loss

        local_loss = local_loss / (num_pos_elem*num_classes)

    net_loss = local_loss / batch_size

    return net_loss,num_i1_ss,num_i2_ss,den_i1_ss,den_i2_ss

            
            

        