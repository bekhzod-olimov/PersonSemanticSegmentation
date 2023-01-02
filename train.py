import cv2, argparse, torch, sys, yaml, os
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import *

def run(args):
    
    # Get the training arguments
    backbone = args.backbone
    epochs = args.epochs
    device = args.device
    path = args.ims_path
    bs = args.batch_size
    lr = args.learning_rate    
    save_path = args.save_path
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    # Read the data
    df = pd.read_csv(path)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Split the data into train, validation sets
    trainset = SegmentationDataset(df=train_df, augmentations=get_train_augs())
    validset = SegmentationDataset(df=valid_df, augmentations=get_valid_augs())
    print(f"Size of Trainset : {len(trainset)}")
    print(f"Size of Validset : {len(validset)}")
    
    # Create train and validation dataloaders
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
    validloader = DataLoader(validset, batch_size=bs, shuffle=False)
    print(f"Number of batches in the trainloader: {len(trainloader)}")
    print(f"Number of batches in the validloader: {len(validloader)}")
    
    # Create the model and move it to gpu
    model = SegmentationModel()
    model.to(device)

    # Train function
    def train_fn(data_loader, model, optimizer):

        model.train()
        total_loss = 0

        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits, loss = model(images, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)


    def eval_fn(data_loader, model):

        model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(data_loader):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                optimizer.zero_grad()
                logits, loss = model(images, masks)
                total_loss += loss.item()

        return total_loss / len(data_loader)
    
    best_valid_loss = np.Inf
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss = eval_fn(validloader, model)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pt')
            print("Best model is saved!")

        print(f"\nEpoch {epoch+1} is completed!")
        print(f"Train loss: {train_loss:.3f}")
        print(f"Validation loss: {valid_loss:.3f}")   
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Person Semantic Segmentation Arguments')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-is", "--im_size", type=int, default=320, help="Images size")
    parser.add_argument("-d", "--device", type=str, default='cuda:1', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='data.csv', help="Path to the data")
    parser.add_argument("-bb", "--backbone", type=str, default='timm-mobilenetv3_large_100', help="Model name for backbone")
    parser.add_argument("-w", "--weights", type=str, default='imagenet', help="Pretrained weights type")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-3, help="Learning rate value")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs")
    args = parser.parse_args() 
    
    run(args) 
