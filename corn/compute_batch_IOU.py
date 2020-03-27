# -*- coding: utf-8 -*-
"""
@purpose: This file is used for batch computation of Precision and Recall at different IoU threshold.
@input: Add all Model paths to "weights" list.
@output: A matlab file will be created for all Precision and recall values at corresponding IoU thresholds.
Created on Sun Dec 23 03:54:14 2018

@author: shrin
"""
import os
import sys
import numpy as np
import tensorflow as tf
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

# Import sample and model
import corn_2class

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#Load both class data config
#config_3class= corn.CornConfig()
CORN_DIR = os.path.join(ROOT_DIR, "datasets/corn")
#class InferenceConfig(config_3class.__class__):
    # Run detection on one image at a time
 #   GPU_COUNT = 1
  #  IMAGES_PER_GPU = 1
   # DETECTION_MIN_CONFIDENCE = 0.8
#config_3class = InferenceConfig()

config= corn_2class.CornConfig()
CORN_DIR = os.path.join(ROOT_DIR, "datasets/corn")
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
config = InferenceConfig()
config.display()

dataset = corn_2class.CornDataset()
dataset.load_corn(CORN_DIR, "test")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device("/gpu:0"):
    #model_3class = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
    #                          config=config_3class)
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

## Calculate AP on batch of images--------------------------------------------------------------------------------------------------------------------
import scipy.io
import numpy as np

def compute_batch_IOU(dataset, image_ids, iou_threshold=0.5):

    IOUs_all=np.array([], dtype=np.float)
    pred_match_all=np.array([], dtype=np.float)
    gt_match_all=np.array([], dtype=np.float)

    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        r = results[0]
        """Compute Average Precision at a set IoU threshold (default 0.5).

        Returns:
        mAP: Mean Average Precision
        precisions: List of precisions at different class score thresholds.
        recalls: List of recall values at different class score thresholds.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        """
        # Get matches and overlaps
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            iou_threshold)

        match_idx=np.where(gt_match>=0)[0].astype(np.int)
        gt_match_idx=gt_match[match_idx].astype(np.int)
        IOUs=overlaps[gt_match_idx, match_idx]
        IOUs_all=np.concatenate((IOUs_all, IOUs), axis=0)
        pred_match_all=np.concatenate((pred_match_all, pred_match), axis=0)
        gt_match_all=np.concatenate((gt_match_all, gt_match), axis=0)
        print("current image is "+ str(image_id))
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match_all > -1) / (np.arange(len(pred_match_all)) + 1)
    recalls = np.cumsum(pred_match_all > -1).astype(np.float32) / len(gt_match_all)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, IOUs_all

# Set path to corn weights file
#weights_path_3class = "C:/Users/shrin/Documents/GitHub/Corn_Detection/logs/appr2/3class_50im_600ep/mask_rcnn_corn_0600.h5"
weights = []
#logs 50 image
weights.append("/home/ssa49593/Mask_RCNN/logs/50im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/50im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/50im_3/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/50im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/home/ssa49593/Mask_RCNN/logs/50im_5/mask_rcnn_corn_2class_0600.h5")

#logs 100 image
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/100im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/home/ssa49593/Mask_RCNN/logs/100im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/100im_3/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/100im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/100im_5/mask_rcnn_corn_2class_0600.h5")

#logs 150 image
weights.append("/work/cylilab/Mask_RCNN/logs/150im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/150im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/150im_3/mask_rcnn_corn_2class_0600.h5")
weights.append("/home/ssa49593/Mask_RCNN/logs/150im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/150im_5/mask_rcnn_corn_2class_0600.h5")

#logs 200 image
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/200im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/200im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/200im_3/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/200im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/200im_5/mask_rcnn_corn_2class_0600.h5")

#logs 250 image
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/250im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/250im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/250im_5/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/250im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/appr1/2class_250im_600ep/mask_rcnn_corn_2class_0600.h5")

#logs 300 logs
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/300im_1/mask_rcnn_corn_2class_0600.h5")
weights.append("/scratch/ssa49593/workDir/Corn_detection/logs/300im_2/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/300im_4/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/appr1/2class_200im_600ep/mask_rcnn_corn_2class_0600.h5")
weights.append("/work/cylilab/Mask_RCNN/logs/appr1/2class_200im_600ep/mask_rcnn_corn_2class_0600.h5")

for weights_path in weights:
    # Load weights
    #print("Loading weights ", weights_path_3class)
    #model_3class.load_weights(weights_path_3class, by_name=True)
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    
    limit=25
    AP_all=[]
    IOU_threshold = []
    presions_recalls_all=[]
    IOUs_all=[]
    
    for i in range(0,5):
        threshold=round(0.7+i*0.05, 2)
        AP, precisions, recalls, IOUs = compute_batch_IOU(dataset, dataset.image_ids[:limit], threshold)
        
        presions_recalls=np.vstack((precisions, recalls))
        AP_all.append(AP)
        IOU_threshold.append(threshold)
        presions_recalls_all.append(presions_recalls)
        IOUs_all.append(IOUs)
    
    len(weights_path)
    filename = weights_path[0:len(weights_path)-29] + "PRCurve.mat"    
    scipy.io.savemat(filename, mdict={'IOU_threshold': IOU_threshold, 'AP': AP_all, 'precision_recall': presions_recalls_all, 'IOUs':IOUs_all})
    break
