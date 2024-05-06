# EOandAIWorkshop

This repository contains Python modules for training and domain adapting any Encoder-Decoder Networks (EDNs) using the `segmentation_models_pytorch` library. It includes comprehensive training and adaptation scripts with customisable settings for various segmentation tasks.

## Key Takeaways

1. Set up a DL environment in Google Drive.
2. Use Python Programs, Google Colab, and free GPU resources.
3. Download and extract building footprints (polygons) of Massachusetts, Boston.
4. Train an image segmentation model based on U-Net architecture with lightweight CNNs like [MobileOne](https://openaccess.thecvf.com/content/CVPR2023/papers/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.pdf) CNN backbone from Apple.
5. Evaluate the model, visualise predictions, and make interpretations
6. Address the small geospatial dataset problem using:
    * Transfer learning
    * Data augmentation
    * Automated data pipeline with API services
  
## Python Modules

This GitHub provides several custom Python modules to help you with smooth workflow in the tutorial provided in Jupyter Notebook `Train-SMpytorch.ipynb` inside `notebook` folder. The modules are:

1. `custom_data_loader`: a custom Dataset function to read images, apply augmentation and preprocessing transformations
2. `data_preprocessing`: creates a pipeline of augmentation strategies for training data
3. `binary_focal_loss_smp`: an implementation of Focal Loss with smooth label cross entropy which is proposed in 'Focal Loss for Dense Object Detection' (https://arxiv.org/abs/1708.02002)
4. `model_fetcher`: fetches a segmentation model, an optimizer, and a loss function
5. `model_trainer`: trains a specified segmentation model with provided configurations
6. `model_evaluate`: evaluates a segmentation model with provided configurations
7. `model_prediction_visualiser`: visualizes predictions using a pre-trained model by displaying the original image along with its ground truth and predicted segmentation masks
8. `model_adapt_transfer_learning`: performs domain adaptation for a specified segmentation model with provided configurations
9. `model_trainer_with_augmentation`: trains a specified segmentation model with provided configurations and augmentations


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

## Usage

### Training a Model

To train a model, use the `train_edns` function from the `model_trainer.py`:
```bash
from model_trainer import train_edns

train_edns(model_name='Unet', cnn_name='resnet34', data_type='building-data',
           CLASSES=['building'], BATCH_SIZE=8, EPOCHS=100, LR=0.001,
           optimiser='Adam', loss_func='DiceLoss', ckpt_name='best_model.pth')
```

### Domain Adaptation of a Model

For domain adaptation, use the `adapt_edns` function from the `model_adapt_transfer_learning.py`:
```bash
from model_adapt_transfer_learning import adapt_edns

adapt_edns(model_name='Unet', bb='resnet34', t_weights_path='path_to_weights.pth',
           data_type='Massachussets Data', CLASSES=['building'], BATCH_SIZE=8,
           EPOCHS=50, LR=0.001, optimiser='Adam', loss_func='DiceLoss', ckpt_name='adapted_model.pth')
```

### Training a Model with Augmentation

To train a model, use the `train_edns_with_augmentation` function from the `model_trainer_with_augmentation.py`:
```bash
from model_trainer_with_augmentation import train_edns_with_augmentation

train_edns_with_augmentation(model_name='Unet', cnn_name='resnet34', data_type='building-data',
           CLASSES=['building'], BATCH_SIZE=8, EPOCHS=100, LR=0.001,
           optimiser='Adam', loss_func='DiceLoss', ckpt_name='best_model.pth')
```

### Detailed Jupyter Notebooks for easy usage

There are two Jupyter Notebooks inside `notebook` folder for easy usage of this GitHub. 
1. `Locate_downloader.ipynb` - Upload this notebook to your Google Drive and run the notebook to clone this repo, set up correct folders, and download the dataset and the trained model checkpoints. Run this notebook only for the first time you set up the files in your Google Drive.
2. `Train-SMpytorch.ipynb` - Find this notebook in your recently cloned GitHub folder inside your Google Drive. The notebook is easy to follow and contains all the required codes.


## Dataset

We use the Massachusetts Building dataset as a benchmark dataset for preliminary studies. It is one of the most used datasets for building extraction with CNNs due to its early release. You can find more information about the dataset at https://www.cs.toronto.edu/~vmnih/data/. For experimental consistency and managing memory efficiently, we make use of a smaller subset of the dataset. Originally composed of $1500 \times 1500$ tiles, this subset divides these into smaller $256 \times 256$ tiles. We also reduce the number of training and testing images by factors of 4 and 2, respectively, to decrease computational time for our thorough comparative analysis. However, the number of validation images remains unchanged from the original dataset. Altogether, the distribution of train, test, and validation images in this subset totals 800, 160, and 100, respectively.

## More details

Please view `Train-SMpytorch.ipynb` inside `notebook` folder for more details.

## Contributors

- Sumesh KC (kcsumesh1993@gmail.com).

## Acknowledgement

Special thanks to the authors and contributors of [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch), without whom the creation of this GitHub would not have been possible.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

We hope this project helps you in your efforts to advance the state of image segmentation using deep learning. For any issues or further inquiries, feel free to reach out through GitHub issues or via email.

Bipul Neupane - geomat.bipul@gmail.com; bneupane@student.unimelb.edu.au
University of Melbourne, Earth Observation and AI Research Group

