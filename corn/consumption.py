# -*- coding: utf-8 -*-
"""
@purpose: This file is used for batch detection of images using all models.
@input: Add all Model paths to "weights" list, and test image directory path "strDirectory".
@output: Masked images along with percent consumption will be saved under output/ directory. A matlab file will be created for all Prediction and Ground Truth values.
Created on Sun Dec 23 03:54:14 2018

@author: shrin
"""

import os
import sys
import numpy as np
import tensorflow as tf
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import save_image
#import 2 different classes
import corn_2class

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

CORN_DIR = os.path.join(ROOT_DIR, "datasets/corn")

config_2class= corn_2class.CornConfig()
CORN_DIR = os.path.join(ROOT_DIR, "datasets/corn")
class InferenceConfig(config_2class.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
config_2class = InferenceConfig()
config_2class.display()

dataset_2class = corn_2class.CornDataset()
dataset_2class.load_corn(CORN_DIR, "test")
dataset_2class.prepare()
print("Images: {}\nClasses: {}".format(len(dataset_2class.image_ids), dataset_2class.class_names))


# Create model in inference mode
with tf.device("/gpu:0"):
    model_2class = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config_2class)
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Set path to corn weights file
#weights_path_2class = os.path.join(ROOT_DIR, "logs/appr1/2class_300im_600ep/mask_rcnn_corn_2class_0600.h5")

# Load weights
#print("Loading weights ", weights_path_3class)
#model_3class.load_weights(weights_path_3class, by_name=True)
#print("Loading weights ", weights_path_2class)
#model_2class.load_weights(weights_path_2class, by_name=True)

def get_cornList(r, n_classes, image) :
    from collections import Counter
    cornList = []
    redCornList=[]
    yellowCornList=[]
    no_of_corns = no_of_red = no_of_yellow = 0
    classes = r['class_ids']
    masks = r['masks']
    regions = r['rois']
    cornMasks = []
    redCornMasks = []
    #print(regions)
    #print(classes)
    #print(masks.shape)
    offset = round(((image.shape)[1])*0.075)
    #print('Offset : ',offset)
    class_detected = Counter(classes)
    no_of_corns = class_detected[1]
    if(n_classes == 2) : 
        no_of_red = class_detected[2]
    elif(n_classes == 3) :
        no_of_yellow = class_detected[2]
        no_of_red = class_detected[3]
    #print(no_of_corns, no_of_red, no_of_yellow)
    for index, roi, class_id in zip(range(len(regions)), regions, classes):
        mask = masks[:,:,index]
        if(class_id == 1):
            #print(mask.shape)
            cornList.append({'cornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum()), 'redCorns' : [], 'yellowCorns' : []})
            cornMasks.append(mask)
        if(class_id == 2 and n_classes == 2) : 
            redCornList.append({'redCornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum())})
            redCornMasks.append(mask)
        elif(class_id == 2 and n_classes == 3) :
            yellowCornList.append({'yellowCornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum())})
        if(class_id == 3 and n_classes == 3) : 
            redCornList.append({'redCornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum())})
    #redCornIdx = []
    for corn in cornList:
        corn_y1 = corn['cornRoi'][0] - offset
        corn_x1 = corn['cornRoi'][1] - offset
        corn_y2 = corn['cornRoi'][2] + offset
        corn_x2 = corn['cornRoi'][3] + offset
        corn_area = corn['mask_pixels']
        eaten_area = 0
       # print('RedCorns Before : ', corn['redCorns'])
        for redCorn in redCornList:
            if((corn_y1 <= redCorn['redCornRoi'][0]) and (corn_x1 <= redCorn['redCornRoi'][1])
            and (corn_y2 >= redCorn['redCornRoi'][2]) and (corn_x2 >= redCorn['redCornRoi'][3])):
               corn['redCorns'].append(redCorn)
               eaten_area += redCorn['mask_pixels']
               #redCornIdx.append(redCorn)
               #redCornList.remove(redCorn)
        percent_eaten = round((eaten_area / corn_area) * 100 , 3)
        corn.update({'percent_eaten' : percent_eaten}) 
        #print('RedCorns After : ', corn['redCorns'])
    #redCornList = [e for e in redCornList if e not in redCornIdx]
#    if len(redCornList) > 0 :
#        print("There are ", len(redCornList) ," undetected corn cob present which are almost fully consumed.")
    #print('RedCorns After : ', redCornList)
    #print('Final CORNS : \n', cornList)    
    leftCorn={}
    left_idx = len(cornList) - 1
    if(len(cornList) > 1):
        for corn_idx in range(len(cornList)):
            corn = cornList[corn_idx]
            corn_y1 = corn['cornRoi'][0]
            corn_x1 = corn['cornRoi'][1]
            corn_y2 = corn['cornRoi'][2]
            corn_x2 = corn['cornRoi'][3]
            height = corn_x2 - corn_x1
            width = corn_y2 - corn_y1
            replaceLeft = False
            if(len(leftCorn) == 0):
                replaceLeft = True
            else :
                if(height > width) :
                    if(corn_y1 < leftCorn['cornRoi'][0]) :
                        replaceLeft = True
                elif(width > height) :
                    if(corn_x1 < leftCorn['cornRoi'][1]) :
                        replaceLeft = True
            if replaceLeft:
                leftCorn = corn
                left_idx = corn_idx
    cornList.pop(left_idx)
    cornList.append(leftCorn)
    if len(cornMasks) > 0:
        ret_cornMasks = np.transpose(np.asarray(cornMasks),(1,2,0))
    else:
        ret_cornMasks = cornMasks
    if len(redCornMasks) > 0:
        ret_redCornMasks = np.transpose(np.asarray(redCornMasks),(1,2,0))
    else:
        ret_redCornMasks = redCornMasks
    return cornList, ret_cornMasks, ret_redCornMasks

def compute_percent_est_accuracy(gt_percent_est, pred_percent_est, thresh):
    if (gt_percent_est - thresh) <= pred_percent_est <= (gt_percent_est + thresh) :
        error = 0 
    elif(gt_percent_est > pred_percent_est):
        error = (gt_percent_est - thresh) - pred_percent_est
    elif(gt_percent_est < pred_percent_est):
        error = (gt_percent_est + thresh) - pred_percent_est
    return (100 - math.fabs(error))

def compare_corns(cornList):
    if cornList[0]['percent_eaten'] < cornList[1]['percent_eaten']:
        return 1
    else:
        return 0

def compare_performance(gt_corns, pred_corns, left_eaten_count):
    #make percent acc calculations
    percent_est_accuracy = 0
    for gt_corn, pred_corn in zip(gt_corns, pred_corns):
       #make percent acc calculations
       est_accuracy = compute_percent_est_accuracy(gt_corn['percent_eaten'], pred_corn['percent_eaten'], thresh=1.0)
       pred_corn.update({'est_accuracy' : est_accuracy})
       percent_est_accuracy += est_accuracy
    percent_est_accuracy = percent_est_accuracy / len(pred_corns)
    
    #make left vs right predictions
    comparison_accuracy = 0
    if(len(gt_corns) > 1 and len(pred_corns) > 1):
        gt_left_eaten_more = compare_corns(gt_corns)
        #print('gt_left_eaten_more : ' , gt_left_eaten_more)
        pred_left_eaten_more = compare_corns(pred_corns)
        #print('pred_left_eaten_more : ' , pred_left_eaten_more)
        if(pred_left_eaten_more == 1) :
            #print('Left corn has been eaten more than Right.')
            left_eaten_count += 1
        #else:
            print('Right corn has been eaten more than Left.')
        if(gt_left_eaten_more == pred_left_eaten_more):
            comparison_accuracy = 1
    else:
        comparison_accuracy = 1
        
    return percent_est_accuracy, left_eaten_count, comparison_accuracy
    

def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    mean_weight_iou = []
    for image_id in image_ids:
        try:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset, config_2class,
                                       image_id, use_mini_mask=False)
            # Run object detection
            results = model_2class.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
            # Compute AP over range 0.5 to 0.95
            r = results[0]
            visualize.save_image(image, "test"+str(image_id), r['rois'], r['masks'],
                    r['class_ids'],r['scores'],['BG', 'Whole Corn','Bare Cob'],scores_thresh=0.8,mode=0, captions=None, show_mask=True)
            gt_r = {"class_ids": gt_class_id,
                   "rois": gt_bbox,
                    "masks": gt_mask}
            gt_corns, gt_corn_masks, gt_red_corn_masks = get_cornList(gt_r, 2, image)
               # print('gt_mask size:  ',gt_corn_masks.shape)
            pred_corns, pred_cornMasks, pred_redCornMasks = get_cornList(r, 2, image)
            #print(pred_corns)
            print(image_id, "Image" , os.path.basename(dataset_2class.source_image_link(image_id)))
            print(image_id, 'percent_eaten_gt', gt_corns[1]['percent_eaten'])
            print(image_id, 'percent_eaten_pred', pred_corns[1]['percent_eaten'])
            print(image_id, 'percent_eaten_gt', gt_corns[0]['percent_eaten'])
            print(image_id, 'percent_eaten_pred', pred_corns[0]['percent_eaten'])
            print("*****************************************************************")
            images.append(os.path.basename(dataset_2class.source_image_link(image_id)))
            gt_one.append(gt_corns[1]['percent_eaten'])
            pred_one.append(pred_corns[1]['percent_eaten'])
            gt_two.append(gt_corns[0]['percent_eaten'])
            pred_two.append(pred_corns[0]['percent_eaten'])
        except:
            print("image Id :", image_id)
            print(sys.exc_info())
            ap = 0
            APs.append(ap)
            if verbose:
                info = dataset.image_info[image_id]
                meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
                print("{:3} {}   AP: {:.2f}".format(
                    meta["image_id"][0], meta["original_image_shape"][0], ap))
            pass
    return APs

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
import scipy.io
import numpy as np
# Run on test set
for weights_path in weights:
    # Load weights
    #print("Loading weights ", weights_path_3class)
    #model_3class.load_weights(weights_path_3class, by_name=True)
    images = []
    gt_one = []
    pred_one = []
    gt_two = []
    pred_two = []
    #weights_path = os.path.join(ROOT_DIR, "logs/appr1/2class_050im_600ep/50im_2/mask_rcnn_corn_2class_0600.h5")
    print("Loading weights ", weights_path)
    model_2class.load_weights(weights_path, by_name=True)
    APs = compute_batch_ap(dataset_2class, dataset_2class.image_ids[5:6])
    filename = weights_path[0:len(weights_path)-29] + "PRCurve.mat"    
    scipy.io.savemat(filename, mdict={'ImageIds': images, 'GT_Left': gt_one, 'Pred_Left': pred_one, 'GT_Right': gt_two, 'Pred_Right': pred_two})
    break