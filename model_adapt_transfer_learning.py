import os, cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter
import albumentations as albu

from model_fetcher import *
from custom_data_loader import Dataset
from data_preprocessing import *
from binary_focal_loss_smp import BinaryFocalLoss

### Helper function for Domain Adaptation
def adapt_edns(model_name, bb, t_weights_path, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_name):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This program performs domain adaptation for a specified segmentation model with provided configurations.

    Parameters:
        model_name (str): Name of the model to adapt.
        bb (str): Backbone of the model, usually a type of pre-trained encoder. 
        t_weights_path (str): Path to the pre-trained weights to load into the model.
        data_type (str): Type of data (e.g., 'building', 'Massachussets Data') used to specify directory structure.
        CLASSES (list of str): List of class names for the segmentation task.
        BATCH_SIZE (int): Batch size for training and validation data loaders.
        EPOCHS (int): Number of training epochs.
        LR (float): Learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use.
        loss_func (str): Name of the loss function to use.
        ckpt_name (str): File name to save the model checkpoint.

    Returns:
        None: This function trains the model and saves the best model based on IoU score.
    """
    ENCODER = bb
    ENCODER_WEIGHTS = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save = model_name + ENCODER
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'

    # tensorBoard writer
    log_directory = './runs/'
    writer = SummaryWriter(comment=f'{log_directory}/{model_name}_{ENCODER}')

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
    # Dataset and Data Loader for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #num_workers=12
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### COMPILE MODEL WITH CUSTOM WEIGHT
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    pretrained_checkpoint = torch.load(t_weights_path)
    pretrained_state_dict = pretrained_checkpoint['model_state_dict']
    model.load_state_dict(pretrained_state_dict)
    model.to(DEVICE)
    model.eval()

    ####### Getting network parameters
    model_params=model.parameters()
    params = sum(p.numel() for p in model.parameters())

    ####### Printing some elements
    print("Training model ", model_name)
    print("Encoder: ", ENCODER)
    print("Network params", params)

    ####### Evaluation metrics
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_loss(loss_func) # collecting loss function
    optim = get_optim(optimiser, model_params, LR) # collecting optimiser
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max') # learning rate scheduler

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optim,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, EPOCHS):

        print('\nEpoch: {}/{}'.format(i+1,EPOCHS))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        # log the loss to TensorBoard
        writer.add_scalar('Loss/Train', train_logs['dice_loss'], i)
        writer.add_scalar('Loss/Valid', valid_logs['dice_loss'], i)

        # log the IoU and F1 score to TensorBoard
        writer.add_scalar('Accuracy/Train', train_logs['iou_score'], i)
        writer.add_scalar('Accuracy/Valid', valid_logs['iou_score'], i)
        writer.add_scalar('Accuracy/Train', train_logs['fscore'], i)
        writer.add_scalar('Accuracy/Valid', valid_logs['fscore'], i)

        # print the final loss and metrics in validation data after each epoch 
        print("Loss: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
            list(valid_logs.items())[0][1],
            list(valid_logs.items())[1][1],
            list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1],
            list(valid_logs.items())[4][1])
        )

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, ckpt_name) # saving a model along with the weights
            print('Model saved!')
            scheduler.step(max_score)

    # close the TensorBoard writer after the loop ends to ensure all data is flushed to disk
    writer.close()
    print("Adaptation Completed!")

