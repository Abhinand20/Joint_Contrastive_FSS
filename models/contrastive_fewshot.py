"""
Defining the models used for training, both models share the same backbone
    1) PANet + ALP module 
    2) UNet for contrastive learning
"""
from collections import OrderedDict
from distutils.log import debug
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.models.unet import DynamicUnet
import torchvision

from .alpmodule import MultiProtoAsConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv'  # using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class conv_bn_relu(nn.Module):
    """
    Standard 2D conv layer
    """
    def __init__(self, in_channels, out_channels, bn=True,use_bias=True,use_relu=True):
        super(conv_bn_relu, self).__init__()
        self.BN_ = bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same", bias=use_bias)
        if self.BN_:
          self.bn = nn.BatchNorm2d(out_channels)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_:
          x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

## Network A -> Define U-Net architecture and use the same backbone for Network B
class ContrastiveBranch(nn.Module):
    def __init__(self,encoder_pre_path=None, no_filters = 16,pretrained_path=None,cfg=None): 
        """
        encoder_pre_path : For encoder
        pretrained_path : For the entire model
        cfg: Some config params (TBD)
        """
        super(ContrastiveBranch,self).__init__()
        self.pretrained_path = pretrained_path
        self.no_filters = no_filters
        self.encoder_pre_path = encoder_pre_path
        self.config = cfg
        self.unet = DynamicUnet(encoder = self.get_shared_encoder(), n_out = no_filters, img_size = (256,256), norm_type=None)
        ## 1x1 conv layer
        self.conv1 = conv_bn_relu(in_channels=no_filters,out_channels=no_filters,bn=True,use_bias=False,use_relu=True)
        self.conv2 = conv_bn_relu(in_channels=no_filters,out_channels=no_filters,bn=False,use_bias=False,use_relu=False)
        

    def get_shared_encoder(self):

        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True, num_classes=21, aux_loss=None)
            _model_list = list(_model.children())
            shared_encoder = _model_list[0]
            print("#### Using DeepLabv3 weights ####")
            
        elif self.config['which_model'] == 'densecl_res101':
            _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None)
            _model_list = list(_model.children())
            shared_encoder = _model_list[0]
            print("#### Using DenseCL weights ####")
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.encoder_pre_path:
            shared_encoder.load_state_dict(torch.load(self.encoder_pre_path))
            print(f'###### Pre-trained encoder f{self.encoder_pre_path} has been loaded ######')

        named_seq = nn.Sequential()
        for idx,module in enumerate(list(shared_encoder.children())):
            named_seq.add_module("shared_{}".format(idx),module)

        return named_seq
        

    def forward(self,x_in):
        fts_out = self.unet(x_in)
        y_out = self.conv1(fts_out)
        y_out_fin = self.conv2(y_out)

        return y_out_fin # (2,H,W,16) 

## Network B -> Takes the shared backbone module as input
class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, shared_encoder ,no_filters = 16,in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.encoder = TVDeeplabRes101Encoder(shared_encoder=shared_encoder,use_coco_init=self.config['use_coco_init'])
        self.get_cls()
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')
    

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.encoder(imgs_concat, low_level = False)
        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1): # batch dimension, fixed to 1
            fg_masks = [] # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']


            scores          = []
            assign_maps     = []
            bg_sim_maps     = []
            fg_sim_maps     = []
        
            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode = BG_PROT_MODE, thresh = BG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), mode = FG_PROT_MODE if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps    = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps    = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps

    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH )

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))