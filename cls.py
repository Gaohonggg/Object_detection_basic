import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights

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

                if self.count_objects_in_annotation(annotation_path) <= 1:
                    valid_image_files.append( img_name )
                
        return valid_image_files
    
    def parse_annotation(self, annotation_path):
        tree = ET.parse( annotation_path )
        root = tree.getroot()
        
        label = None
        for obj in root.findall("object"):
            name = obj.find("name").text
            if label is None:
                label = name
            
        tag = ["cat","dog"]
        try:
            return tag.index( label )
        except ValueError:
            return -1
    
    def __len__(self):
        return len( self.image_files )
    
    def __getitem__(self, index):
        img_name = self.image_files[ index ]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")

        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.annotations_dir, annotation_name)

        label = self.parse_annotation( annotation_path )

        if self.transform:
            image = self.transform( image )
        
        return image, label
    

if __name__ == "__main__":
    data_dir = "data"
    annotations_dir = os.path.join(data_dir, "annotations")
    image_dir = os.path.join(data_dir, "images")

    image_files = [
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(
            image_dir, f
        )) 
    ]

    df = pd.DataFrame({"image_name": image_files})

    train_df, val_df = train_test_split(df,
                                        test_size=0.2,
                                        random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDataset(annotations_dir, 
                                 image_dir,
                                 transform)
    val_dataset = ImageDataset(annotations_dir,
                               image_dir,
                               transform)
    
    train_dataset.image_files = [
        f for f in train_dataset.image_files if f in train_df["image_name"].values
    ]
    val_dataset.image_files = [
        f for f in val_dataset.image_files if f in val_df["image_name"].values
    ]

    train_loader = DataLoader(train_dataset,
                              batch_size= 32,
                              shuffle= True)
    val_loader = DataLoader(val_dataset,
                            batch_size= 32,
                            shuffle= False)
    

    model = models.resnet18(weights= ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to( device) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to( device )
            targets = targets.to( device )

            scores = model( data )
            loss = criterion( scores, targets )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for data, targets in val_loader:
                data = data.to( device )
                targets = targets.to( device )

                scores = model( data )
                _, predictions = scores.max(1)

                correct += (predictions == targets).sum()
                total += targets.size(0)
            
            print(f"Epoch {epoch+1}, Validation Accuracy: {float(correct)/float(total)*100:.2f}%")

























