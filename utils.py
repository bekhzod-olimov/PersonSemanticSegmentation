# Import libraries
import cv2, torch
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A

# Set train variables
image_size = 320
weights = "imagenet"

# Train transformations function
def get_train_augs():
          
          """

          This function initialize and return train dataset transformations.

          Output:

               train process transformations.

          """

          return A.Compose([
                 A.Resize(image_size, image_size),
                 A.HorizontalFlip(p = 0.5),
                 A.VerticalFlip(p = 0.5)
          ])

def get_valid_augs():
          
      """
      
      This function initialize and return validation dataset transformations.
      
      Output:
      
           train process transformations.
      
      """
          
      return A.Compose([A.Resize(image_size, image_size)])

def get_imgs_and_masks(row):
          
    """
    
    Gets row in the dataframe manipulates images and masks;
    returns image and mask.
    
    Argument:
    row - row of the dataframe.
    
    """
    # Get image and mask
    image_path = row.images
    mask_path = row.masks
    
    # Image and mask manipulation
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image_size, image_size))
    mask = np.expand_dims(cv2.resize(mask, (image_size, image_size)), axis = -1)

    return image, mask

class SegmentationDataset(Dataset):
    
          """

          This function gets a dataframe and augmentations and returns a dataset.

          Arguments:

                df            - a dataframe, pandas object;
                augmentations - transformations.
                
          Output:
          
                dataset       - dataset, torch data object.

          """

    # Initialization
    def __init__(self, df, augmentations): self.df, self.augmentations = df, augmentations

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
                    
        row = self.df.iloc[idx]
        image, mask = get_imgs_and_masks(row)
        
        # Apply augmentations
        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32)

        image = torch.Tensor(image) / 255.
        mask = torch.round(torch.Tensor(mask) / 255.)

        return image, mask

# Model class
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        # Get Unet with pretrained weights
        self.arc = smp.Unet(
            encoder_name = encoder,
            encoder_weights = weights,
            in_channels = 3,
            classes = 1,
            activation = None
        )

    def forward(self, images, masks=None):
        
        # Get predicted masks
        logits = self.arc(images)

        # Compute two loss values
        if masks != None:
          loss1 = DiceLoss(mode='binary')(logits, masks)
          loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            
            return logits, loss1 + loss2

        return logits
