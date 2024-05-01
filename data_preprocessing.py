import albumentations as albu

def get_training_augmentation():
    """
    Source: https://github.com/qubvel/segmentation_models.pytorch/tree/master

    This function creates a pipeline of augmentation strategies for training data.
    
    This pipeline includes:
    - Horizontal flipping
    - Scaling, shifting, and rotating
    - Padding and random cropping to a fixed size
    - Adding Gaussian noise
    - Applying perspective transformations
    - Adjusting contrast using various methods
    - Adjusting sharpness or blurring
    
    Returns:
        A callable composed of all the augmentations.
    """
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=280, min_width=280, always_apply=True, border_mode=0),
        albu.RandomCrop(height=256, width=256, always_apply=True),

        albu.transforms.GaussNoise(p=0.2),
        albu.geometric.transforms.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                #albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                #albu.IAASharpen(p=1),
                albu.transforms.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """
    Create a pipeline of augmentation strategies for validation data.
    
    This function specifically adds padding to make the image shape divisible by 32, which is often
    required by neural network architectures for consistent input dimensions.
    
    Returns:
        A callable composed of the padding augmentation.
    """
    test_transform = [
        albu.PadIfNeeded(280, 280)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert images to tensor format for model input.
    
    Parameters:
        x (numpy.ndarray): The image data in height x width x channels format.
    
    Returns:
        numpy.ndarray: The image data in channels x height x width format, as a float32.
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """
    Construct preprocessing transform.
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

