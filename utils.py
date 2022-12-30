import cv2, torch
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A

IMAGE_SIZE = 320
ENCODER = "timm-mobilenetv3_large_100"
WEIGHTS = "imagenet"
BATCH_SIZE = 16
DEVICE = 'cuda:1'

def get_train_augs():
          return A.Compose([
              A.Resize(IMAGE_SIZE, IMAGE_SIZE),
              A.HorizontalFlip(p=0.5),
              A.VerticalFlip(p=0.5)
          ])

def get_valid_augs():
      return A.Compose([
          A.Resize(IMAGE_SIZE, IMAGE_SIZE),
  ])

def get_imgs_and_masks(row):

    image_path = row.images
    mask_path = row.masks
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    mask = np.expand_dims(cv2.resize(mask, (IMAGE_SIZE,IMAGE_SIZE)), axis=-1)

    return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, mask = get_imgs_and_masks(row)
        # print(image.shape)
        # print(mask.shape)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32)

        image = torch.Tensor(image) / 255.
        mask = torch.round(torch.Tensor(mask) / 255.)

        return image, mask
    
    
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.arc = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks != None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            
            return logits, loss1 + loss2

        return logits
