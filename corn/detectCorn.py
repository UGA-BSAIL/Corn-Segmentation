# -*- coding: utf-8 -*-
"""
@purpose: This file is used for batch detection of images.
@input: Change Model path "weights_path_2class", and test image directory path "strDirectory".
@output: Masked images along with percent consumption will be saved under output/ directory.
Created on Sun Dec 23 03:54:14 2018

@author: shrin
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib

import corn_2class

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#Load both class data config
config_2class= corn_2class.CornConfig()
CORN_DIR = os.path.join(ROOT_DIR, "datasets/corn")
class InferenceConfig(config_2class.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
config_2class = InferenceConfig()
config_2class.display()

# Load validation dataset
#dataset_2class = corn_2class.CornDataset()
#dataset_2class.load_corn(CORN_DIR, "val")
#dataset_2class.prepare()
#print("Images: {}\nClasses: {}".format(len(dataset_2class.image_ids), dataset_2class.class_names))

# Create model in inference mode
with tf.device("/gpu:0"):
    model_2class = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config_2class)

weights_path_2class = "C:/Users/shrin/Documents/GitHub/Corn_Detection/logs/appr1/2class_300im_600ep/300im_1/mask_rcnn_corn_2class_0600.h5"

# Load weights
print("Loading weights ", weights_path_2class)
model_2class.load_weights(weights_path_2class, by_name=True)

def get_cornList(r, n_classes) :
    from collections import Counter
    cornList = []
    redCornList=[]
    yellowCornList=[]
    no_of_corns = no_of_red = no_of_yellow = 0
    classes = r['class_ids']
    masks = r['masks']
    regions = r['rois']
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
    print(no_of_corns, no_of_red, no_of_yellow)
    for index, roi, class_id in zip(range(len(regions)), regions, classes):
        mask = masks[:,:,index]
        if(class_id == 1):
            #print(mask.shape)
            cornList.append({'cornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum()), 'redCorns' : [], 'yellowCorns' : []})
        if(class_id == 2 and n_classes == 2) : 
            redCornList.append({'redCornRoi' : roi, 'class_id' : class_id, 'mask' : mask, 'mask_pixels' : (mask.sum())})
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
    return cornList
#strDirectory = "D:/test data/Corn Test Data/5- 95 consumed/"
strDirectory = "C:/Users/shrin/Documents/GitHub/Corn_Detection/datasets/corn/newTest/"
directory = os.fsencode(strDirectory)
count = 1
for file in os.listdir(directory):
    try:
        r2  = {}
        filename = os.fsdecode(file)
        print(str(count) + ": Image Under Process: ", filename)
    #    image = skimage.io.imread("D:/test data/Corn Test Data/5- 95 consumed/1275_1_00404_01248_a1.JPG")
        image = skimage.io.imread(strDirectory + str(filename))
        results_2class = model_2class.detect([image], verbose=0)
        r2 = results_2class[0]
        visualize.save_image(image, filename, r2['rois'], r2['masks'],
                    r2['class_ids'],r2['scores'],['BG', 'Corn','Red Corn Kernel'],scores_thresh=0.8,mode=0, captions=None, show_mask=True)
        #print('Detection results: ' ,r2)
        cornList_2class  = get_cornList(r2, 2)
        final_rois = []
        final_masks = []
        final_class = []
        final_percent = []
        cornList = cornList_2class
        for corn, index in zip(cornList, range(len(cornList))) :
                print("Corn number ", index+1, "is  ",corn['percent_eaten'],"% Eaten")
                final_rois.append(corn['cornRoi'])
                final_masks.append(corn['mask'])
                final_class.append(corn['class_id'])
                final_percent.append((str(corn['percent_eaten'])+'% Eaten'))
                for redCorn in corn['redCorns'] :
                    final_rois.append(redCorn['redCornRoi'])
                    final_masks.append(redCorn['mask'])
                    final_class.append(redCorn['class_id'])
                    final_percent.append('')
        final_rois = np.asarray(final_rois)
        final_masks = np.asarray(final_masks)
        final_class = np.asarray(final_class)
        shape = final_masks.shape
        final_masks.shape = (shape[1], shape[2], shape[0])
        if len(cornList) > 0:
            visualize.save_image(image, filename+"result", final_rois, final_masks,
                    final_class,r2['scores'],['BG', 'Corn','Red Corn Kernel'],scores_thresh=0.8,mode=0, captions=final_percent, show_mask=False)
    except Exception as execp:
        print('exception occurred for image ', filename)
        print('Details are : ', execp)
        continue
    finally:
#        image = None
#        r2 = {}
#        cornList_2class = cornList= final_rois = final_masks = final_class = final_percent = None
        count += 1
#        if count > 1:
#           break