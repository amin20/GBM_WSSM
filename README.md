# GBM_WSSM

- Model Name: GBM_WSSM
- "Glioblastoma Whole-Slide Image Semantic Segmentation Model using Deep Convolutional Neural Network"

# -----------------------------------------------

<img width="1004" alt="Fig_3" src="https://user-images.githubusercontent.com/35130196/88754186-2c359080-d19d-11ea-9fbb-523db587cd8b.png">

# -----------------------------------------------

# Dependencies: 

- Python: 3.6
- Tensorflow: 1.13.1
- Keras: 2.2.4

# -----------------------------------------------


# Steps to reproduce the results of the paper

1. Download the optimum weights of "[GBM_WSSM](https://github.com/amin20/GBM_WSSM)" trained on IVY-GAP dataset from the link "[GBM_WSSM.h5](https://1drv.ms/u/s!AvQTr_2MktoOgQT_ZMbotiquQW0L?e=ehFJah)" and place it in its appropriate address in the file "[GBM_WSSM_Prediction.py](https://github.com/amin20/GBM_WSSM/tree/master/Codes)".

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

2. Download GBM test images from this [link](https://github.com/amin20/GBM_WSSM/tree/master/GBM_Test_Images) or use your own GBM images. For pre-processing of your images, please refer to the original article. Then, run the file "[GBM_WSSM_Prediction.py](https://github.com/amin20/GBM_WSSM/tree/master/Codes)". you can compare the results (predicted masks) with the original ground truth, check this [link](https://github.com/amin20/GBM_WSSM/tree/master/GBM_Test_Ground_Truth).

3. If you want to train the model from scratch, download raw images and corressponding masks from the link . Also, you can save your time and train the model by using training and validating numpy arrays we prepared. For both ways, first download the images or arrays from and put them in the [main_code](https://github.com/amin20/GBM_WSSM/blob/master/Codes/0_main.py).

Arguments:

    GBM_Parser.add_argument("--GBM_Images_Path",
                            help = "The path of the original GBM images",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--GBM_GTs_Path",
                            help = "The path of the original GBM Ground Truth (GTs)",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--GBM_Regions_and_RGBs_List",
                            help = "A text file which defines all the regions (classes) inside GBM GTs along with their RGB vlaues i.e. Mask_Labels.txt",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--Encode_GBM_GTs",
                            help = "The default value is False. If you change it to True, all the GTs images will be mapped (RGBs will be encoded to integers) using Mask_Integer_Encoding()",
                            default = True)
    
    GBM_Parser.add_argument("--Model_Outputs_Path",
                            help = "In this path, all the model outputs will be saved (including: data arrays, curves, weights.hdf5, ...",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--Log_Files_Path",
                            help = "This path will save callbacks outputs: checkpoint files (*.h5 or *.hdf5 formats), TensorBoard records, and events from CSVLogger (*.csv)",
                            default = ".../.../")
    
    GBM_Parser.add_argument("--Hyper_parameters",
                            help = "A list that contains all necessary hyperparameters: Learning_Rate, Epochs, Patience, LR_Decay, Train_Percentage, Batch_Size, and class no., respectively",
                            default = [1e-3, 55, 50, 5e-5, 85, 1, 8])
    
    GBM_Parser.add_argument("--GBM_Image_Size",
                            help = "The new size of GBM images. The size can be used if you want to use Resize function. Default is (1024, 1024)",
                            default = (1024, 1024))
   
    GBM_Parser.add_argument("--Image_Size_Boolean",
                            help = "Image_Size_Boolean is an argument whcih is being used whenever we need to resize the originl images or GTs",
                            default = True)
    
    GBM_Parser.add_argument("--Cropped_Size",
                            help = "Crop_Size is used as a predefined size for crop extraction in Augmentaion procedure. Default is (224, 224)",
                            default = (224,224))
    
    GBM_Parser.add_argument("--Resume",
                            help = "The default value is False, If you change it to True, the training process will be resumed provided you saved the weights", 
                            default = False)
    
    GBM_Parser.add_argument("--Start_from_Scratch",
                            help = "If you are going to start from scratch i.e. make up training, and validating sets, and encode all masks, and train the model, apply --> TRUE. Otherwise, if you apply False, just predefined arrays are loaded from disk and the model will be trained on", 
                            default = False)

# -----------------------------------------------

# Steps to generalize GBM_WSSM on TCGA dataset

1. Original Images (.SVS format) for all types of brain tumours are accessible through the link. 

2. We extracted only GBM tumour types using the [GBM Log File] and [GBM Finder Code] GBM images (.SVS format) are accessible through the link.

3. All GBM tumours were visually checked by a pathologist and whole slide images (WSIs) were extracted for each GBM. They are acceible via the [link].

4. Resized form (1024x1024) of the GBM WSIs are available in this [link].  

5. Corresponding produced masks (segmented masks) by applying GBM_WSSM on GBM WSIs are accessible in the [link]. 

6. We applyed the [code] to quantify regions in segmented masks obtained from previous step. The quantified values are located [here].

# -----------------------------------------------

# Notes

DCNN Network Architecture inspired by the [Article](https://arxiv.org/abs/1611.09326) 
