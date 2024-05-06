import os, cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from model_fetcher import *
from custom_data_loader import Dataset
from data_preprocessing import *
from binary_focal_loss_smp import BinaryFocalLoss
import matplotlib.pyplot as plt


# helper function for data visualization
def visualiser(**images):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function displays images with their respective ground truth and predicted masks in a single row.

    Parameters:
        images (dict): A dictionary containing 'image', 'ground_truth_mask', and 'predicted_mask' data.
    
    No return value; this function directly plots the images using matplotlib.
    """
    fig = plt.figure(figsize=(16, 5))
    image, gt_mask, pr_mask = images.get('image'), images.get('ground_truth_mask'), images.get('predicted_mask')

    # invert
    inverted_gt_mask = 1 - gt_mask
    inverted_pr_mask = 1 - pr_mask

    # show image
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(image, 'gray', interpolation='none')
    ax1.set_title('Original Image')
    ax1.set_axis_off()

    # show gt
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(image, 'gray', interpolation='none')
    ax2.imshow(inverted_gt_mask, 'bone', interpolation='none', alpha=0.6)
    ax2.set_title('Ground truth label')
    ax2.set_axis_off()

    # show pr
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(image, 'gray', interpolation='none')
    ax3.imshow(inverted_pr_mask, 'bone', interpolation='none', alpha=0.6)
    ax3.set_title('Prediction from model')
    ax3.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()


def visualise_predictions(saved_ckpt, edn, bb, data_type, loss_func, image_idx):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function allows to visualize predictions using a pre-trained model by displaying the original image along with its ground truth
    and predicted segmentation masks.

    Parameters:
        saved_ckpt (str): Path to the saved model checkpoint.
        edn (str): Model name used in the segmentation model.
        bb (str): Backbone (encoder) of the model.
        data_type (str): Type of data to specify the directory for loading images.
        loss_func (str): Loss function used for model evaluation.
        image_idx (int): Index of the image to visualize from the validation dataset.
    
    No return value; this function loads the model, processes an image from the validation dataset,
    predicts the mask, and calls the visualiser to display the results.
    """
    ENCODER = bb
    CLASSES = ['building']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = None
    ACTIVATION = 'sigmoid'
    print("********************************************************************************")
    print("********************************************************************************")

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    # Dataset for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### Load model, load pretrained Weights, and add into model
    model = get_model_from_smp(edn, bb, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    print("Encoder: ", bb)
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()

    ###### VISUALISE on val dataset without transformations for image visualization
    valid_dataset_vis = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES,)

    n = image_idx
    image_vis = valid_dataset_vis[n][0].astype('uint8')
    image, gt_mask = valid_dataset[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualiser(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)

    print("********************************************************************************")
    print("********************************************************************************")


def visualise_predictions_five_image(saved_ckpt, edn, bb, data_type, loss_func):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function allows to visualize predictions using a pre-trained model by displaying five original images along with their ground truth
    and predicted segmentation masks.

    Parameters:
        saved_ckpt (str): Path to the saved model checkpoint.
        edn (str): Model name used in the segmentation model.
        bb (str): Backbone (encoder) of the model.
        data_type (str): Type of data to specify the directory for loading images.
        loss_func (str): Loss function used for model evaluation.
        image_idx (int): Index of the image to visualize from the validation dataset.
    
    No return value; this function loads the model, processes an image from the validation dataset,
    predicts the mask, and calls the visualiser to display the results for 5 randomly selected images.
    """
    ENCODER = bb
    CLASSES = ['building']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = None
    ACTIVATION = 'sigmoid'
    print("********************************************************************************")
    print("********************************************************************************")
    
    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'   
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    # Dataset for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### Load model, load pretrained Weights, and add into model
    model = get_model_from_smp(edn, bb, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    print("Encoder: ", bb)
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()
    
    ###### VISUALISE
    valid_dataset_vis = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES,)
    for i in range(5):
        n = np.random.choice(len(valid_dataset))

        image_vis = valid_dataset_vis[n][0].astype('uint8')
        image, gt_mask = valid_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualiser(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)
    
    print("********************************************************************************")
    print("********************************************************************************")
