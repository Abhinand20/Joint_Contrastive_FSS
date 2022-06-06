import os
import os.path as osp
import random
import numpy as np
import torch
from torch.nn import DataParallel
import config
from ssn import SSN
from bsds500 import BSDS500, transform_patch_data_train, transform_convert_label
from evaluation import ASA
from torch.utils.data import DataLoader
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import skimage

def segment_colorfulness(image, mask):
	# split the image into its respective RGB components, then mask
	# each of the individual RGB channels so we can compute
	# statistics only for the masked region
	(B, G, R) = cv2.split(image.astype("float"))
	R = np.ma.masked_array(R, mask=mask)
	G = np.ma.masked_array(B, mask=mask)
	B = np.ma.masked_array(B, mask=mask)
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`,
	# then combine them
	stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
	meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

torch.cuda.set_device(config.DEVICE)
model_dir = 'ssn_pytorch/saved_models/latest.pth'

val_transform = transform_convert_label(config.NUM_CLASSES)

ValDataset = BSDS500(root = config.ROOT, split = 'val', transform = val_transform)
ValLoader = DataLoader( ValDataset,
                        batch_size = 1,
                        #num_workers = config.NUM_WORKERS,
                        shuffle = False,
                        drop_last= False)

net = SSN(config.IN_CHANNELS, config.OUT_CHANNELS, config.C, config.K_MAX, config.ITER_EM, config.COLOR_SCALE, config.P_SCALE, config.TRAIN_H, config.TRAIN_W, config.NUM_CLASSES).cuda()
#net = DataParallel(net, device_ids = config.DEVICES)
net.eval()



for i, val_data in enumerate(ValLoader):
    #net.module.reset_xylab_Estep(config.K_MAX_VAL, ValDataset.height, ValDataset.width)
    net.reset_xylab_Estep(config.K_MAX_VAL, ValDataset.height, ValDataset.width)
    images_lab, labels = val_data
    images_lab, labels = images_lab.cuda(), labels.cuda()
    posteriors = net(images_lab)
    superpixel = ASA(posteriors, labels, config.NUM_CLASSES, config.K_MAX)

    #superpixel = superpixel.transpose(1,2,0)
    superpixel = superpixel[0]
    images = np.float32(images_lab[0,...].permute(1,2,0).cpu())
    orig = skimage.img_as_ubyte(images)
    vis = np.zeros(orig.shape[:2], dtype="float")


    # compute the colorfulness of each superpixel
    for v in np.unique(superpixel):
	    # construct a mask for the segment so we can compute image
	    # statistics for *only* the masked region
	    mask = np.ones(images.shape[:2])
	    mask[superpixel == v] = 0
	    # compute the superpixel colorfulness, then update the
	    # visualization array
	    C = segment_colorfulness(orig, mask)
	    vis[superpixel == v] = C


    vis = skimage.exposure.rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
    # overlay the superpixel colorfulness visualization on the original
    # image
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = orig.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.imwrite('{}_superpixel.jpg'.format(i), output)
    cv2.imwrite("{}_Visualization.jpg".format(i), vis)
    cv2.imwrite('{}.jpg'.format(i), orig)
    print('save image')
    
    del posteriors
    torch.cuda.empty_cache()








