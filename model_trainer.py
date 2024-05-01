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

### Helper function to train an EDN with an encoder of CNN
def train_edns(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_name):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This program trains a specified segmentation model with provided configurations.

    Parameters:
        model_name (str): Name of the model to adapt.
        bb (str): Backbone (encoder CNN) of the model, usually a type of pre-trained encoder. 
        data_type (str): Type of data (e.g., 'building-data', 'Mass-small') used to specify directory structure.
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

    # tensorBoard writer
    log_directory = './runs/'
    writer = SummaryWriter(comment=f'{log_directory}/{model_name}_{ENCODER}')

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
    # Dataset and Data Loader for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    # create segmentation model with pretrained encoder
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    model_params=model.parameters()
    params = sum(p.numel() for p in model.parameters())

    ####### PRINTING SOME DETAILS
    print("Encoder: ", ENCODER)
    print("Dataset: ", data_type)
    print("Network params: ", params)

    ####### METRICS AND HYPERPARAMETER SETUP
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model_params, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')
    max_score = 0

    ####### SETUP TRAINING EPOCH
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optim,
        device=DEVICE,
        verbose=True,
    )

    ####### SETUP VALIDATION EPOCH
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    ####### TRAINING AND VALIDATION LOOP
    for i in range(EPOCHS):
        print('\nEpoch: {}/{}'.format(i+1, EPOCHS))
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
        print("Loss: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, IoU: {:.3f}, F1: {:.3f}".format(
            list(valid_logs.items())[0][1],
            list(valid_logs.items())[1][1],
            list(valid_logs.items())[2][1],
            list(valid_logs.items())[3][1],
            list(valid_logs.items())[4][1])
        )

        # saving model only when the IoU score from current epoch is higher than the previous IoU score
        if max_score < valid_logs['iou_score']:
          max_score = valid_logs['iou_score']
          torch.save({'model_state_dict': model.state_dict()}, ckpt_name)
          print('Model saved!')
          scheduler.step(max_score)

    # close the TensorBoard writer after the loop ends to ensure all data is flushed to disk
    writer.close()
    print("Training completed.")