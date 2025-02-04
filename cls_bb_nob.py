import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tqdm as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, annotations_dir, image_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self.filter_images_with_multiple_objects()

    def count_objects_in_annotation(self, annotation_path):
        try:
            tree = ET.parse( annotation_path )
            root = tree.getroot()
            count = 0
            
            for obj in root.findall("object"):
                count += 1
            return count
        
        except FileNotFoundError:
            return 0
    
    def filter_images_with_multiple_objects(self):
        valid_image_files = []
        for f in os.listdir( self.image_dir ):
            if os.path.isfile( os.path.join(self.image_dir, f) ):
                img_name = f
                annotation_name = os.path.splitext(img_name)[0] + ".xml"
                annotation_path = os.path.join(self.annotations_dir, annotation_name)

                if self.count_objects_in_annotation(annotation_path) == 1:
                    valid_image_files.append( img_name )
                
        return valid_image_files
    
    def parse_annotation(self, annotation_path):
        tree = ET.parse( annotation_path )
        root = tree.getroot()

        image_w = int( root.find("size/width").text )
        image_h = int( root.find("size/height").text )
        
        label = None
        bbox = None
        for obj in root.findall("object"):
            name = obj.find("name").text
            if label is None:
                label = name
                xmin = int( obj.find("bndbox/xmin").text )
                ymin = int( obj.find("bndbox/ymin").text )
                xmax = int( obj.find("bndbox/xmax").text )
                ymax = int( obj.find("bndbox/ymax").text )

                bbox = [
                    xmin / image_w,
                    ymin / image_h,
                    xmax / image_w,
                    ymax / image_h
                ]
            
        tag = ["cat","dog"]
        try:
            return tag.index( label ), torch.tensor(bbox, dtype=torch.float32)
        except ValueError:
            return -1
    
    def __len__(self):
        return len( self.image_files )
    
    def __getitem__(self, index):
        img_1_name = self.image_files[ index ]
        img_1_path = os.path.join(self.image_dir, img_1_name)
        
        image_1 = Image.open(img_1_path).convert("RGB")

        annotation_name = os.path.splitext(img_1_name)[0] + ".xml"
        img_1_annotation_path = os.path.join(self.annotations_dir, annotation_name)
        img_1_annotation_path = self.parse_annotation( img_1_annotation_path )

        index_2 = random.randint(0, len(self.image_files) - 1 )
        img_2_name = self.image_files[ index_2 ]
        img_2_path = os.path.join(self.image_dir, img_2_name)
        
        image_2 = Image.open(img_2_path).convert("RGB")

        annotation_name = os.path.splitext(img_2_name)[0] + ".xml"
        img_2_annotation_path = os.path.join(self.annotations_dir, annotation_name)
        img_2_annotation_path = self.parse_annotation( img_2_annotation_path )

        merged_img = Image.new(
            "RGB", (image_1.width + image_2.width, max(image_1.height, image_2.height))
        )
        merged_img.paste(image_1, (0,0))
        merged_img.paste(image_2, (image_1.width, 0))
        merged_w = image_1.width + image_2.width
        merged_h = max(image_1.height, image_2.height)

        merged_annotations = []
        merged_annotations.append(
            {"bbox": img_1_annotation_path[1].tolist(),
             "label": img_1_annotation_path[0]}
        )

        new_bbox_2 = [
            (img_2_annotation_path[1][0] * image_2.width + image_1.width) / merged_w,
            img_2_annotation_path[1][1] * image_2.height / merged_h,
            (img_2_annotation_path[1][2] * image_2.width + image_1.width) / merged_w,
            img_2_annotation_path[1][3] * image_2.height / merged_h,
        ]

        merged_annotations.append(
            {"bbox": new_bbox_2,
             "label": img_2_annotation_path[0]}
        )

        if self.transform:
            merged_img = self.transform( merged_img )
        else:
            merged_img = transforms.ToTensor()( merged_img )

        annotations = torch.zeros( (len(merged_annotations), 5) )
        for i, ann in enumerate(merged_annotations):
            annotations[i] = torch.cat(
                (torch.tensor(ann["bbox"]), torch.tensor([ann["label"]]))
            )

        return merged_img, annotations

class TwoHeadedModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.resnet50(weights= ResNet50_Weights.DEFAULT)
        self.num_classes = num_class

        self.backbone = nn.Sequential(
            *list( self.base_model.children() )[:-2]
        )
        self.fcs = nn.Linear(
            2048, 2*2*(4 + self.num_classes)
        )

    def forward(self, x):
        x = self.backbone( x )
        x = F.adaptive_avg_pool2d( x, (1,1))
        x = x.view( x.size(0), -1)
        x = self.fcs( x )
        return x
    

def calculate_loss(output, targets, device, num_classes):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    batch_size = output.shape[0]
    total_loss = 0
    
    # Reshape to (batch_size, grid_y, grid_x, 4 + num_classes)
    output = output.view(batch_size, 2, 2, 4 + num_classes)
    
    for i in range(batch_size):  # Iterate through each image in the batch
        for j in range(len(targets[i])):  # Iterate through objects in the image
            # Determine which grid cell the object's center falls into
            bbox_center_x = (targets[i][j][0] + targets[i][j][2]) / 2
            bbox_center_y = (targets[i][j][1] + targets[i][j][3]) / 2
            
            grid_x = int(bbox_center_x * 2)  # Multiply by number of grid cells (2 in this case)
            grid_y = int(bbox_center_y * 2)
            
            # Classification Loss for the responsible grid cell
            label_one_hot = torch.zeros(num_classes, device=device)
            label_one_hot[int(targets[i][j][4])] = 1
            classification_loss = ce_loss(output[i, grid_y, grid_x, 4:], label_one_hot)
            
            # Regression Loss for the responsible grid cell
            bbox_target = targets[i][j][:4].to(device)
            regression_loss = mse_loss(output[i, grid_y, grid_x, :4], bbox_target)
            
            # No Object Loss (for other grid cells)
            no_obj_loss = 0
            for other_grid_y in range(2):
                for other_grid_x in range(2):
                    if other_grid_y != grid_y or other_grid_x != grid_x:
                        no_obj_loss += mse_loss(
                            output[i, other_grid_y, other_grid_x, :4],
                            torch.zeros(4, device=device)
                        )
            
            total_loss += classification_loss + regression_loss + no_obj_loss
    
    return total_loss / batch_size  # Average loss over the batch

def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            images = images.to(device)
            output = model(images)
            
            total_loss = calculate_loss(output, targets, device, num_classes)
            running_loss += total_loss.item()
            
            # Reshape output to (batch_size, grid_y, grid_x, 4 + num_classes)
            output = output.view(images.shape[0], 2, 2, 4 + num_classes)
            
            # Collect predictions and targets for mAP calculation
            for batch_idx in range(images.shape[0]):
                for target in targets[batch_idx]:
                    bbox_center_x = (target[0] + target[2]) / 2
                    bbox_center_y = (target[1] + target[3]) / 2
                    grid_x = int(bbox_center_x * 2)
                    grid_y = int(bbox_center_y * 2)
                    
                    # Class prediction (index of max probability)
                    prediction = output[batch_idx, grid_y, grid_x, 4:].argmax().item()
                    all_predictions.append(prediction)
                    all_targets.append(target[4].item())
    
    val_loss = running_loss / len(data_loader)
    
    # Convert lists to tensors for PyTorch’s metric functions
    all_predictions = torch.tensor(all_predictions, device=device)
    all_targets = torch.tensor(all_targets, device=device)
    
    # Calculate accuracy
    val_accuracy = (all_predictions == all_targets).float().mean()
    
    return val_loss, val_accuracy.item()

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, num_classes):
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        
        for images, targets in tqdm.tqdm(train_loader, desc="Batches", leave=False):
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            total_loss = calculate_loss(output, targets, device, num_classes)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, device, num_classes)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
    
    return train_losses, val_losses, train_accuracies, val_accuracies
    

if __name__ == "__main__":
    data_dir = "data"
    annotations_dir = os.path.join(data_dir, "annotations")
    image_dir = os.path.join(data_dir, "images")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(annotations_dir, image_dir, transform)
    train_dataset, val_dataset = train_test_split(dataset, 
                                                  test_size= 0.2,
                                                  random_state= 42)
    train_loader = DataLoader(train_dataset,
                              batch_size= 8,
                              shuffle= True)
    val_loader = DataLoader(val_dataset,
                            batch_size= 8,
                            shuffle= False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    class_to_idx = {"dog":0, "cat":1}

    model = TwoHeadedModel(num_classes).to( device )
    optimizer = optim.Adam( model.parameters(), lr= 0.001 )
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model= model,
        train_loader= train_loader,
        val_loader= val_loader,
        optimizer= optimizer,
        num_epochs= 10,
        device= device,
        num_classes= num_classes
    )