import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from binary_focal_loss_smp import BinaryFocalLoss

def get_model_from_smp(model_name, enc_name, enc_weight, num_class, act):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function fetches a segmentation model from the segmentation_models_pytorch library with specified configurations.

    Parameters:
        model_name (str): The name of the segmentation model.
        enc_name (str): The name of the encoder to use in the model.
        enc_weight (str or None): Pre-trained weights for the encoder or None to initialize weights randomly.
        num_class (int): The number of classes for segmentation.
        act (str): Activation function to use in the output layer of the model.

    Returns:
        A segmentation model configured as specified. Prints an error message if the model name is incorrect.
    """
    print('Model: ', model_name)
    input_shape=(256, 256, 3)
    if model_name == 'DeepLabV3':
        return_model = smp.DeepLabV3(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'DeepLabV3Plus':
        return_model = smp.DeepLabV3Plus(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'FPN':
        return_model = smp.FPN(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'Linknet':
        return_model = smp.Linknet(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'MAnet':
        return_model = smp.MAnet(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'PAN':
        return_model = smp.PAN(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'PSPNet':
        return_model = smp.PSPNet(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'Unet':
        return_model = smp.Unet(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    elif model_name == 'UnetPlusPlus':
        return_model = smp.UnetPlusPlus(encoder_name=enc_name, encoder_weights=enc_weight, classes=num_class, activation=act)
    else:
        return_model = print('Model name is wrong.')
    return return_model


def get_optim(opt, model_params, LR):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function creates an optimizer for training based on the specified type and parameters.

    Parameters:
        opt (str): The name of the optimizer (e.g., 'Adam', 'SGD').
        model_params (iterable): Parameters of the model to optimize.
        LR (float): Learning rate for the optimizer.

    Returns:
        An instantiated optimizer. Prints an error message if the optimizer name is incorrect.
    """
    print('Optimiser: ', opt)
    input_shape=(256, 256, 3)
    if opt == 'Adam':
        optim = torch.optim.Adam([ dict(params=model_params, lr=LR),])
    elif opt == 'SGD':
        optim = torch.optim.SGD([ dict(params=model_params, lr=LR),])
    elif opt == 'RMSprop':
        optim = torch.optim.RMSprop([ dict(params=model_params, lr=LR),])
    elif opt == 'Adadelta':
        optim = torch.optim.Adadelta([ dict(params=model_params, lr=LR),])
    elif opt == 'Adagrad':
        optim = torch.optim.Adagrad([ dict(params=model_params, lr=LR),])
    elif opt == 'Adamax':
        optim = torch.optim.Adamax([ dict(params=model_params, lr=LR),])
    elif opt == 'NAdam':
        optim = torch.optim.NAdam([ dict(params=model_params, lr=LR),])
    else:
        optim = print('Optimiser name is wrong.')
    return optim


def get_loss(loss_func):
    """
    Prepared by: Bipul Neupane (geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au)

    This function retrieves a loss function based on the specified name, potentially combining different losses.

    Parameters:
        loss_func (str): The name of the loss function or a combination thereof.

    Returns:
        The requested loss function. Prints an error message if the loss function name is incorrect.
    """
    print('Loss Function: ', loss_func)
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    jaccard_loss = smp.utils.losses.JaccardLoss()
    dice_loss = smp.utils.losses.DiceLoss()
    binary_crossentropy = smp.utils.losses.BCELoss()
    binary_focal_loss = BinaryFocalLoss()

    # loss function combinations
    bce_dice_loss =  binary_crossentropy + dice_loss
    bce_jaccard_loss = binary_crossentropy + jaccard_loss
    binary_focal_dice_loss = binary_focal_loss + dice_loss
    binary_focal_jaccard_loss = binary_focal_loss + jaccard_loss
    total_loss = dice_loss + binary_focal_loss

    if loss_func == 'jaccard_loss':
        return_lf = jaccard_loss
    elif loss_func == 'dice_loss':
        return_lf = dice_loss
    elif loss_func == 'binary_focal_loss':
        return_lf = binary_focal_loss
    elif loss_func == 'binary_crossentropy':
        return_lf = binary_crossentropy
    elif loss_func == 'bce_dice_loss':
        return_lf = bce_dice_loss
    elif loss_func == 'bce_jaccard_loss':
        return_lf = bce_jaccard_loss
    elif loss_func == 'binary_focal_dice_loss':
        return_lf = binary_focal_dice_loss
    elif loss_func == 'binary_focal_jaccard_loss':
        return_lf = binary_focal_jaccard_loss
    elif loss_func == 'total_loss':
        return_lf = total_loss
    else:
        return_lf = print('Loss name is wrong.')
    return return_lf