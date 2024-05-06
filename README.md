# EOandAIWorkshop

This repository contains Python modules for training and domain adapting any Encoder-Decoder Networks (EDNs) using the `segmentation_models_pytorch` library. It includes comprehensive training and adaptation scripts with customizable settings for various segmentation tasks.

## Features

- Training EDNs with customisable backbones and decoders.
- Domain adaptation capabilities for transferring learned models to new, distinct datasets.
- Integration with TensorBoard for real-time training and validation metrics visualization.
- Supports various loss functions and optimizers for robust model training.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch 1.7 or higher
- segmentation_models_pytorch
- matplotlib
- albumentations

You can install the necessary libraries with pip:
```bash
pip install torch torchvision segmentation-models-pytorch albumentations matplotlib
```

Please see the other requirements at https://github.com/qubvel/segmentation_models.pytorch/blob/master/requirements.txt


### Installation
```
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

### Usage

#### Training a Model

To train a model, use the `train_edns` function from the `model_trainer.py`:
```bash
from model_trainer import train_edns

train_edns(model_name='Unet', cnn_name='resnet34', data_type='building-data',
           CLASSES=['building'], BATCH_SIZE=8, EPOCHS=100, LR=0.001,
           optimiser='Adam', loss_func='DiceLoss', ckpt_name='best_model.pth')
```

#### Domain Adaptation of a Model

For domain adaptation, use the `adapt_edns` function from the `model_adapt_transfer_learning.py`:
```bash
from model_adapt_transfer_learning import adapt_edns

adapt_edns(model_name='Unet', bb='resnet34', t_weights_path='path_to_weights.pth',
           data_type='Massachussets Data', CLASSES=['building'], BATCH_SIZE=8,
           EPOCHS=50, LR=0.001, optimiser='Adam', loss_func='DiceLoss', ckpt_name='adapted_model.pth')
```

#### Training a Model with Augmentation

To train a model, use the `train_edns_with_augmentation` function from the `model_trainer_with_augmentation.py`:
```bash
from model_trainer_with_augmentation import train_edns_with_augmentation

train_edns_with_augmentation(model_name='Unet', cnn_name='resnet34', data_type='building-data',
           CLASSES=['building'], BATCH_SIZE=8, EPOCHS=100, LR=0.001,
           optimiser='Adam', loss_func='DiceLoss', ckpt_name='best_model.pth')
```

#### Detailed Jupyter Notebooks for easy usage

There are two Jupyter Notebooks inside `notebook` folder for easy usage of this GitHub. 
1. `Locate_downloader.ipynb` - Upload this notebook to your Google Drive and run the notebook to prepare clone this repo, setup correct folders, download the dataset and trained model checkpoints. Run this notebook only for the first time you setup the files in your Google Drive.
2. `Train-SMpytorch.ipynb` - Find this notebook in your recently cloned GitHub folder inside your Google Drive. The notebook is easy to follow and contains all the required codes.


### Dataset

We use the Massachusetts Building dataset as a benchmark dataset for preliminary studies. It is one of the most used datasets for building extraction with CNNs due to its early release. You can find more information about the dataset at https://www.cs.toronto.edu/~vmnih/data/. For experimental consistency and managing memory efficiently, we make use of a smaller subset of the dataset. Originally composed of $1500 \times 1500$ tiles, this subset divides these into smaller $256 \times 256$ tiles. We also reduce the number of training and testing images by factors of 4 and 2, respectively, to decrease computational time for our thorough comparative analysis. However, the number of validation images remains unchanged from the original dataset. Altogether, the distribution of train, test, and validation images in this subset totals 800, 160, and 100, respectively.

### More details.

Please view `Train-SMpytorch.ipynb` inside `notebook` folder for more details.

### Contributors

Special thanks to Contributor Sumesh KC (mailto:kcsumesh1993@gmail.com).

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contact

Bipul Neupane - geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au

