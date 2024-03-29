# Import libraries
import cv2, torch
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A

# Set train variables
image_size, weights = 320, "imagenet"

# Train transformations function
def get_train_augs():
          
    """

    This function initialize and return train dataset transformations.

    Output:

         train process transformations, albumentations compose object.

    """

    return A.Compose([ A.Resize(image_size, image_size), A.HorizontalFlip(p = 0.5), A.VerticalFlip(p = 0.5) ])

def get_valid_augs():
          
    """
    
    This function initialize and return validation dataset transformations.
    
    Output:
    
         validation process transformations, albumentations compose object.
    
    """
        
    return A.Compose([A.Resize(image_size, image_size)])

def get_imgs_and_masks(row):
          
    """
    
    This function gets row in the dataframe manipulates images and masks; returns image and mask.
    
    Parameter:
    
          row   - a row of the dataframe.
          
    Outputs:
    
          image - an output image, array;
          mask  - corresponding mask of the image, array.
    
    """

    # Get image and mask
    image_path, mask_path = row.images, row.masks
    
    # Image and mask manipulation
    image, mask = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image, mask = cv2.resize(image, (image_size, image_size)), np.expand_dims(cv2.resize(mask, (image_size, image_size)), axis = -1)

    return image, mask

class SegmentationDataset(Dataset):
    
    """

    This class gets a dataframe and augmentations and returns a dataset.

    Parameters:

          df            - a dataframe, pandas dataframe object;
          augmentations - transformations, albumentations compose object.
          
    Output:
    
          dataset       - dataset, torch dataset object.

    """

    # Initialization
    def __init__(self, df, augmentations): self.df, self.augmentations = df, augmentations

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
                    
        """
        
        This function gets an index in the dataset and reads, applies several functions and returns an image with a corresponding mask.
        
        Parameter:
          
               idx    - index of an image path in the dataset, int.
               
        Outputs:
        
               image  - an output image, tensor;
               mask   - an output mask, tensor.
        
        """

        # Get data
        row = self.df.iloc[idx]

        # Get an image and its corresponding mask
        image, mask = get_imgs_and_masks(row)
        
        # Apply augmentations
        if self.augmentations:
            data = self.augmentations(image = image, mask = mask)
          
            # Get a transformed image and its corresponding transformed mask
            image, mask = data["image"], data["mask"]
        
        # Change array to tensor
        image, mask = np.transpose(image, (2, 0, 1)).astype(np.float32), np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # Normalize and return image and its corresponding mask
        return torch.Tensor(image) / 255., torch.round(torch.Tensor(mask) / 255.)

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        # Get Unet with pretrained weights
        self.arc = smp.Unet( encoder_name = encoder, encoder_weights = weights, in_channels = 3, classes = 1, activation = None )

    def forward(self, images, masks = None):
        
        # Get predicted masks
        logits = self.arc(images)

        # Compute two loss values
        if masks != None:
          loss1 = DiceLoss(mode='binary')(logits, masks)
          loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            
            return logits, loss1 + loss2

        return logits
