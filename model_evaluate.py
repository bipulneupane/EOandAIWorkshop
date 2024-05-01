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

def evaluate_edns(saved_ckpt, edn, bb, data_type, loss_func):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This program evaluates a segmentation model with provided configurations.

    Parameters:
        saved_ckpt (str): Name of the saved model (.pth file)
        edn (str): Name of the model to adapt.
        bb (str): Backbone of the model, usually a type of pre-trained encoder. 
        data_type (str): Type of data (e.g., 'building', 'Massachussets Data') used to specify directory structure.
        loss_func (str): Name of the loss function used.

    Returns:
        None: This function evaluates the saved model and displays the loss and evaluation metrics.
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
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())

    loss = get_loss(loss_func)
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    test_epoch = smp.utils.train.ValidEpoch(model=model,loss=loss,metrics=metrics,device=DEVICE,verbose=True,)

    ####### PRINTING SOME DETAILS
    print("Encoder: ", bb)
    print("Checkpoint: ", saved_ckpt)
    print("Validated on: ", data_type)
    print("Net Params: ", params)

    valid_logs = test_epoch.run(valid_dataloader) # on valid folder

    print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}".format(
        list(valid_logs.items())[0][1],
        list(valid_logs.items())[1][1],
        list(valid_logs.items())[2][1],
        list(valid_logs.items())[3][1],
        list(valid_logs.items())[4][1])
         )

    print("********************************************************************************")
    print("********************************************************************************")