# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:34:58 2019

@author: Keshik
"""

from __future__ import division

from utils import non_max_suppression, bbox_iou_numpy, compute_ap
import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import gc
import torch
import os

def train(model, 
          device, 
          optimizer, 
          scheduler, 
          train_dataloader, 
          csv_log_file,
          tensor_type=torch.cuda.FloatTensor,
          update_gradient_samples = 16, 
          freeze_darknet=False):
    """
        Train a deep neural yolo network model for a single epoch 
        
        Args
            model: pytorch model object
            device: cuda or cpu
            optimizer: pytorch optimizer object
            scheduler: learning rate scheduler object that wraps the optimizer
            train_dataloader: training  images dataloader
            save_dir (string): Location to save model weights, plots and log_files
            model_num: Integer identifier for the model
            tensor_type: cuda float/ cpu float
            update_gradient_samples (int): number of samples the network sees before updating gradients
            freeze_darknet (bool): If true freeze backbone
    
    """
    
    model.train(True)
    scheduler.step()
    
    if freeze_darknet:
        print("Frezezing backbone...")
        for i, (name, p) in enumerate(model.named_parameters()):
            if int(name.split('.')[1]) < 75:  # if layer < 75
                p.requires_grad = False
    else:
        for i, (name, p) in enumerate(model.named_parameters()):
            if int(name.split('.')[1]) < 75:  # if layer < 75
                p.requires_grad = True
                    
   
    # set gradients to zero
    optimizer.zero_grad()
    
    for i, (img_paths, images, labels) in enumerate(tqdm.tqdm(train_dataloader)):
        images = Variable(images.type(tensor_type))
        labels = Variable(labels.type(tensor_type), requires_grad=False)
        
        # Calculate loss
        loss = model(images, labels)
        
        # Backpropate
        loss.backward()
        
        # Update gradients after some batches
        if ((i + 1)%update_gradient_samples== 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
        
        # Clear variables from gpu, collect garbage and clear gpu cache memory
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
        
        # Construct loss data
        loss_data = "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f"% (
                model.losses["x"] ,
                model.losses["y"] ,
                model.losses["w"] ,
                model.losses["h"] ,
                model.losses["conf"] ,
                model.losses["cls"] ,
                loss.item(),
                model.losses["recall"] ,
                model.losses["precision"],
            )
        
        # Write batch_loss details to csv file
        #writer = csv.writer(csv_log_file)
        csv_log_file.writerow(loss_data.split("\t"))
        
    print(
        "[Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"% (
            model.losses["x"],
            model.losses["y"],
            model.losses["w"],
            model.losses["h"],
            model.losses["conf"],
            model.losses["cls"],
            loss.item(),
            model.losses["recall"],
            model.losses["precision"],
        )
    )
    
    return model




def validation(model, 
          device, 
          valid_dataloader, 
          csv_log_file,
          tensor_type=torch.cuda.FloatTensor,
          num_classes = 8):
    
    # Set model to evaluation mode
    model.train(False)
    model.eval()
    
    # Variables to store detections
    all_detections = []
    all_annotations = []
    
    
    for batch_i, (_, images, labels) in enumerate(tqdm.tqdm(valid_dataloader, desc="Detecting objects")):

        images = Variable(images.type(tensor_type))

        with torch.no_grad():
            outputs = model(images)
            outputs = non_max_suppression(outputs, num_classes, conf_thres=0.8, nms_thres=0.4)
            
        for output, annotations in zip(outputs, labels):
            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= 416

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]
        
        # Clear variables from gpu, collect garbage and clear gpu cache memory
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
    
    ap_list = []
    for c, ap in average_precisions.items():
        print(f"+ Class '{c}' - AP: {ap}")
        # Write ap details to csv file
        ap_list.append(ap)
    
    mAP = np.mean(list(average_precisions.values()))
    ap_list.append(mAP)
    print(f"mAP: {mAP}")

    #writer = csv.writer(csv_log_file)
    csv_log_file.writerow(ap_list)
    
    return model, mAP


def train_model(model, 
          device, 
          optimizer, 
          scheduler, 
          train_dataloader,
          valid_dataloader,
          csv_log_file_train,
          csv_log_file_valid,
          weights_path,
          max_epochs = 10,
          tensor_type=torch.cuda.FloatTensor,
          update_gradient_samples = 16, 
          freeze_darknet=False,
          freeze_epoch = -1):
    
    best_mAP = 0.0
    
    for i in range(0, max_epochs):
        print("--------Epoch {}--------".format(i+1))
        if (freeze_darknet and freeze_epoch != -1) and (i+1 >= freeze_epoch):
            model = train(model, 
                      device, 
                      optimizer, 
                      scheduler, 
                      train_dataloader, 
                      csv_log_file_train,
                      tensor_type=torch.cuda.FloatTensor,
                      update_gradient_samples = 16, 
                      freeze_darknet=freeze_darknet)
        else:
            model = train(model, 
                      device, 
                      optimizer, 
                      scheduler, 
                      train_dataloader, 
                      csv_log_file_train,
                      tensor_type=torch.cuda.FloatTensor,
                      update_gradient_samples = 16, 
                      freeze_darknet=False)
                
        
        
        model, mAP = validation(model, 
            device, 
            valid_dataloader, 
            csv_log_file_valid,
            tensor_type=torch.cuda.FloatTensor,
            num_classes = 8)
        
        # save casual weights here
        model.save_weights(os.path.join(weights_path, "weights_kitti-epoch-{}.weights".format(i+1)))
        
        # update mAP
        if mAP > best_mAP:
            best_mAP = mAP
            print("Saving best weights")
            model.save_weights(os.path.join(weights_path, "best_weights_kitti.weights"))
    
        
    

