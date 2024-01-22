# Advanced Methods for Image Processing course at the University of Bordeaux
## Weed segmentation final project on the CropSegmentation Dataset

Authors: Hala __Zayzafoun__, Bolutife __Atoki__, Tamas __Bukits__
 
 ## Introduction
The aim of this project was to implement crop and weed segmentation on images taken by a drone
on crop fields, by segmenting each pixel in the image into either the background, a crop, or a weed.
This was performed by implementing networks created across the lab sessions (VAE and UNET
architectures) for segmenting, and applying the Convolutional Blind Denoising Network for noise
removal, and then evaluating the segmentation networks using Metrics provided during the lecture
and labs.

## Installation

To use this project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bukits/weed-segmentation.git
   cd your-repository
   ```

2. **Installing Python packages:**

    The project requieres the use uf GPU, the code was tested on Windows using CUDA 10.2 and on Ubuntu using CUDA 11.8.

    Intsalling the python packages you have to run the package installation file:

    ``` 
    conda env create --file=environment.yml
    ```

## Training the model

1. To train the UNet model, run the following command:
    ``` 
    python train-unet.py
    ```
    which will create the __model_unet.pth__.
2. To train the Variational Auto Encoders model, run the following command:
    ``` 
    python train-vae.py
    ```
    which will create the __model_vae.pth__.

## Testing the model

Testing the model on a selected test dataset you have to run the following command with the paramaeters:

* test_folder: path to the test dataset which has 2 folders (images, labels)
* model_path: path to the trained model which can be UNet or VAE.
* batch_size: you can provide the batch size for testing, the deafult is 8.

```
python run.py −−test_folder $dataset_path$ −−model_path $model_path$
```
After running this command the script generates a metrics.txt file where the used metrics are collected. Also, the script plots 5 randomly generated examples with original image, and the predicted mask. 

## Results