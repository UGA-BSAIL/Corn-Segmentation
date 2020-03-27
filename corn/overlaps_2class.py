# -*- coding: utf-8 -*-
"""
@purpose: This file can be alternatively used for batch computation of overlaps i.e. IoU for all test images.
@input: Add all Model paths to "weights" list.
@output: Text file with both class' mean overlaps for each model.
Created on Sun Dec 23 03:54:14 2018

@author: shrin
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images

#import 2 different classes
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
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Set path to corn weights file
#weights_path_3class = "C:/Users/shrin/Documents/GitHub/Corn_Detection/logs/appr2/3class_50im_600ep/mask_rcnn_corn_0600.h5"
weights_path = os.path.join(ROOT_DIR, "logs/appr_comp/2class_50im_600ep_pq/mask_rcnn_corn_2class_0600.h5")

# Load weights
#print("Loading weights ", weights_path_3class)
#model_3class.load_weights(weights_path_3class, by_name=True)
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = utils.trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = utils.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = utils.compute_overlaps_masks(pred_masks, gt_masks)
    masks1 = pred_masks
    masks2 = gt_masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    
    gt_class_area = np.zeros(3)
    for i, ids in enumerate(gt_class_ids):
        gt_class_area[ids] += area2[i]
    
    
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, gt_class_area, area1, intersections


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps, gt_class_area, pred_class_area, intersections = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

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
    
    
    match_iou = np.zeros([gt_boxes.shape[0]])
    match_intersection = np.zeros([gt_boxes.shape[0]])
    weight_iou = np.zeros(3)
    
    for i, pred_id in enumerate(gt_match):
        match_intersection[i] = intersections[int(pred_id), i]
        match_iou[i] = overlaps[int(pred_id) , i]
    
        weight_iou[gt_class_ids[i]] += match_iou[i] * match_intersection[i] / gt_class_area[gt_class_ids[i]]
    
    for idx in range(len(weight_iou)):
        if weight_iou[idx] == 0 and idx not in gt_class_ids:
            weight_iou[idx] = -1
#    weight_iou[idx for idx in range(len(weight_iou)) if weight_iou[idx] == 0 and idx not in gt_class_ids] = -1
    weight_iou        
    return mAP, precisions, recalls, overlaps, weight_iou

def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    np.insert(iou_thresholds, 0, 0)
    # Compute AP over range of IoU thresholds
    AP = []
    weightd_IoU = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps, weight_iou =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
            print("IoU @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap) 
        weightd_IoU.append(weight_iou[1:])
        
    AP = np.array(AP[1:]).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP, weightd_IoU


def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    mean_weight_iou = []
    for image_id in image_ids:
        try:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset, config,
                                       image_id, use_mini_mask=False)
            # Run object detection
            results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
            # Compute AP over range 0.5 to 0.95
            r = results[0]
            ap, weightd_IoU  = compute_ap_range(
                gt_bbox, gt_class_id, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks'],
                verbose=0)
            APs.append(ap)
            mean_weight_iou.append(weightd_IoU[0])
            if verbose:
                info = dataset.image_info[image_id]
                meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
                print("{:3} {}   AP: {:.2f}".format(
                    meta["image_id"][0], meta["original_image_shape"][0], ap), " and mean IoU for class1 and class2 are : ", weightd_IoU[0][0], weightd_IoU[0][1])
        except:
            print("image Id :", image_id)
            #print(sys.exc_info()[0])
            ap = 0
            APs.append(ap)
            if verbose:
                info = dataset.image_info[image_id]
                meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
                print("{:3} {}   AP: {:.2f}".format(
                    meta["image_id"][0], meta["original_image_shape"][0], ap))
            pass
    return APs, mean_weight_iou


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

    # Run on validation set
    limit = 25
    APs = []
    mean_weight_iou = 0
    APs, mean_weight_iou = compute_batch_ap(dataset, dataset.image_ids[:limit])
    print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))
    class1_iou = np.mean(np.asarray([c1 for c1, c2 in mean_weight_iou if c1 > -1]))
    class2_iou = np.mean(np.asarray([c2 for c1, c2 in mean_weight_iou if c2 > -1]))
    print("Mean IoU for both classes for {} images: {:.4f}, {:.4f}".format(len(APs), class1_iou, class2_iou))