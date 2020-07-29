# GBM_WSSM
Glioblastoma Pathology Images Semantic Segmentation using DCNN

<img width="1004" alt="Fig_3" src="https://user-images.githubusercontent.com/35130196/88754186-2c359080-d19d-11ea-9fbb-523db587cd8b.png">

# Dependencies: 

- Python: 3.6
- Tensorflow: 1.13.1
- Keras: 2.2.4

# Steps to reproduce the results of the paper

1. Download weights (GBM_WSSM.h5) trained on IVY dataset from the link and place it in the file "GBM_WSSM_Prediction.py". See the link [Link](https://github.com/Lydorn/polycnn/edit/master/README.md)

2. Download and setup the "Distributed Solar Photovoltaic Array Location and Extent Data Set for Remote Sensing Object Identification" dataset, see the corresponding [subdirectory](data/photovoltaic_array_location_dataset).
3. Download the pre-trained InceptionV4 checkpoint, see the corresponding [subdirectory](models/inception).
4. Train PolyCNN and run inference on the test set, see the corresponding [subdirectory](code/polycnn).
5. Train the U-Net of unet_and_vectorization and run inference on the test set, see the corresponding  [subdirectory](code/unet_and_vectorization).
5. Compare the two methods, see the corresponding [subdirectory](code/evaluation).
