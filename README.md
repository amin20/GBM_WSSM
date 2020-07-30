# GBM_WSSM
Glioblastoma Pathology Images Semantic Segmentation using DCNN

# -----------------------------------------------

<img width="1004" alt="Fig_3" src="https://user-images.githubusercontent.com/35130196/88754186-2c359080-d19d-11ea-9fbb-523db587cd8b.png">

# -----------------------------------------------

# Dependencies: 

- Python: 3.6
- Tensorflow: 1.13.1
- Keras: 2.2.4

# -----------------------------------------------


# Steps to reproduce the results of the paper

1. Download weights (GBM_WSSM.h5) trained on IVY dataset from the link [GBM_WSSM.h5](https://drive.google.com/file/d/1gMPA9R0zToIzgQdA5Gxos481ucxkDtbS/view?usp=sharing) and place it in its appropriate address in the file "[GBM_WSSM_Prediction.py](https://github.com/amin20/GBM_WSSM/tree/master/Codes)".

Arguments:
    GBM_Parser.add_argument("--Image_Info",
                            help = "This part includes some initial information about the image you want to create mask for: shape, number of classes",
                            default = [(1024, 1024, 3), 8])
    
    GBM_Parser.add_argument("--GBM_Test_Images_Path",
                            help = "The path of 48 GBM images in testing phase",
                            default = ".../.../Test/Images/")
    
    GBM_Parser.add_argument("--Predicted_Masks_Path",
                            help = "The path of the Predicted Masks by GBM_WSSM",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--GBM_WSSM_h5_path",
                            help = "GBM_WSSM best model weights.h5 path",
                            default = ".../.../GBM_WSSM.h5")
    
    GBM_Parser.add_argument("--Labels_txt",
                            help = "Mask_Labels.txt path",
                            default = ".../.../Mask_Labels.txt")

# -----------------------------------------------

2. Download GBM test images from this [address](https://github.com/amin20/GBM_WSSM/tree/master/GBM_Test_Images) or use your own GBM test image. For image preprocessing of your test file please refer to the original article. Then, run the file "[GBM_WSSM_Prediction.py](https://github.com/amin20/GBM_WSSM/tree/master/Codes)". you can compare the results (predicted masks) with the original ground truth, check this [link](https://github.com/amin20/GBM_WSSM/tree/master/GBM_Test_Ground_Truth).

3. If you want to train the model from scratch, download raw images and corressponding masks from the link . Also, you can save your time and train the model by using training and validating numpy arrays we prepared. For both ways, first download the images or arrays from and put them in the [main_code](https://github.com/amin20/GBM_WSSM/blob/master/Codes/0_main.py).

# -----------------------------------------------

# Steps to generalize GBM_WSSM on TCGA dataset

1. Original Images (.SVS format) for all types of brain tumours are accessible through the link. 

2. We extracted only GBM tumour types using the [GBM Log File] and [GBM Finder Code] GBM images (.SVS format) are accessible through the link.

3. All GBM tumours were visually checked by a pathologist and whole slide images (WSIs) were extracted for each GBM. They are acceible via the [link].

4. Resized form (1024x1024) of the GBM WSIs are available in this [link].  
