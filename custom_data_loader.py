import os, cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

class Dataset(BaseDataset):
    """
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['building']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            image_size=(256, 256),  # Specify the desired image size
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        # Added these later
        self.image_size = image_size  # Store the image size
        
        # Add the Resize transformation to your preprocessing pipeline
        if preprocessing is None:
            self.preprocessing = albu.Compose([
                albu.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST),
                # ... (other preprocessing transformations you might have)
            ])
        else:
            self.preprocessing = preprocessing
            self.preprocessing.transforms.insert(
                0,
                albu.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST)
            )
        # Added upto here
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        
        #print(self.images_fps[i]) # printing the name of the image that is giving trouble
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        #print("Mask size before resizing:", mask.shape)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        #print("Mask size after resizing:", mask.shape)
        
        return image, mask
        
    def __len__(self):
        return len(self.ids)