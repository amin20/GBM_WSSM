# Import Required Modules

import os
import gc
import sys
gc.collect()
import keras
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model
Image.MAX_IMAGE_PIXELS = 1000000000
from GBM_WSSM import GBM_WSSM_Model
from Mask_Integer_Encoding import main
from Class_Augmentation import Augmentation
gpu_options = tf.GPUOptions(allow_growth=True)
from Load_Resize_Normalize import Images_or_GTs_Path, Normalize
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from Callbacks_More_Images_Show_Per_Epoch import WeightVisualizerCallback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


##############################################################################

# Parse Arguments

def parse_args(arguments):
    
    GBM_Parser = argparse.ArgumentParser(description = "This part defines all the required variables (command line arguments) needed for Glioblastoma (GBM) Images semantic segmenatation")
    
    # All the necessary arguments (variables) are defined as follow 
    
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
    
    return GBM_Parser.parse_args(arguments)

##############################################################################

# Import all Arguments

All_Arguments = sys.argv[1:]
All_Arguments = parse_args(All_Arguments)

GBM_Images_Path = All_Arguments.GBM_Images_Path
GBM_GTs_Path = All_Arguments.GBM_GTs_Path
GBM_Regions_and_RGBs_List = All_Arguments.GBM_Regions_and_RGBs_List
Encode_GBM_GTs = All_Arguments.Encode_GBM_GTs
Model_Outputs_Path = All_Arguments.Model_Outputs_Path
Log_Files_Path = All_Arguments.Log_Files_Path
GBM_Image_Size = All_Arguments.GBM_Image_Size
Image_Size_Boolean = All_Arguments.Image_Size_Boolean
Cropped_Size = All_Arguments.Cropped_Size
Resume = All_Arguments.Resume
Start_from_Scratch = All_Arguments.Start_from_Scratch
Learning_Rate =  All_Arguments.Hyper_parameters[0]
Epochs = All_Arguments.Hyper_parameters[1]
Patience = All_Arguments.Hyper_parameters[2]
LR_Decay = All_Arguments.Hyper_parameters[3]
Train_Percentage = All_Arguments.Hyper_parameters[4]
Batch_Size = All_Arguments.Hyper_parameters[5]
Class_No = All_Arguments.Hyper_parameters[6]

##############################################################################

if Start_from_Scratch:


    # Load Images and Masks as Arrays + Resize (If is required) + Save on Disk
    
    Original_Images_Arrays = Images_or_GTs_Path(GBM_Images_Path, GBM_Image_Size, Image_Size_Boolean)
    Original_Masks_Arrays =  Images_or_GTs_Path(GBM_GTs_Path, GBM_Image_Size, Image_Size_Boolean)
    
    np.save(Model_Outputs_Path + "Original_Images_Arrays", Original_Images_Arrays)
    np.save(Model_Outputs_Path + "Original_Masks_Arrays", Original_Masks_Arrays)
    
    ##############################################################################
    
    # Normalize Arrays + Save on Disk
    
    Normalized_Images_Arrays = Normalize(Original_Images_Arrays)
    np.save(Model_Outputs_Path + "Normalized_Images_Arrays", Normalized_Images_Arrays)
    
    ##############################################################################
    
    # Mask Encoding to Integers
    
    if Encode_GBM_GTs:
        Encoded_Masks_Arrays = main(Original_Masks_Arrays, GBM_Regions_and_RGBs_List, GBM_Image_Size[0], GBM_Image_Size[1])
        np.save(Model_Outputs_Path + "Encoded_Masks_Arrays", Encoded_Masks_Arrays)
        
    ##############################################################################
    
    # Training & Validating Sets
    
    Total_Samples_No = len(Original_Images_Arrays)
    Training_Samples_No = round(Total_Samples_No * (Train_Percentage / 100))
    
    X_Train = Normalized_Images_Arrays[:Training_Samples_No]
    Y_Train = Encoded_Masks_Arrays[:Training_Samples_No]
    X_Validation = Normalized_Images_Arrays[Training_Samples_No:]
    Y_Validation = Encoded_Masks_Arrays[Training_Samples_No:]
    Y_Validation_for_Show = Original_Masks_Arrays[Training_Samples_No:]
    
##############################################################################

elif not Start_from_Scratch:
    
    X_Train = np.load(".../.../X_Train.npy")
    Y_Train = np.load(".../.../Y_Train.npy")
    X_Validation = np.load(".../.../X_Validation.npy")
    Y_Validation = np.load(".../.../Y_Validation.npy")
    Y_Validation_for_Show = np.load(".../.../Y_Validation_Raw.npy")
    

##############################################################################

# New Defined Callback for Realtime Performance Check During Validating Phase

figPath = os.path.sep.join([Model_Outputs_Path, "{}_Loss.png".format(os.getpid())])
figPath2 = os.path.sep.join([Model_Outputs_Path, "{}_Accuracy.png".format(os.getpid())])
jsonPath = os.path.sep.join([Model_Outputs_Path, "{}.json".format(os.getpid())])

Visualizer = WeightVisualizerCallback(X_Validation[10:11], 
                               Y_Validation_for_Show[10:11], 
                               figPath, 
                               figPath2, 
                               jsonPath, 
                               H=224, 
                               W=224, 
                               patch_size = 224, 
                               figsize=8)

##############################################################################

# To call Augmentation Class

Train_Augmentation = Augmentation(Batch_Size, X_Train, Y_Train, Cropped_Size, Train_Phase = True)
Test_Augmentation  = Augmentation(Batch_Size, X_Validation, Y_Validation, Cropped_Size, Train_Phase = False)

##############################################################################

# GBM_WSSM Training

Input_Shape = (224, 224, 3)
Image_Input = Input(shape = Input_Shape)
Output = GBM_WSSM_Model(Class_No, Image_Input)
my_model = Model(inputs = Image_Input, outputs = Output)
Plot_Model = plot_model(my_model, to_file = Model_Outputs_Path + "GBM_WSSM_Architecture.png", show_shapes = True)


if Resume:
    my_model.load_weights(Model_Outputs_Path + "GBM_WSSM.h5")
    
##############################################################################

# Callbacks 

Optimizer = keras.optimizers.RMSprop(Learning_Rate, decay=LR_Decay)

my_model.compile(loss ='sparse_categorical_crossentropy',
              optimizer = Optimizer, 
              metrics = ["accuracy"])


Tensor_Board = TensorBoard(log_dir=Log_Files_Path)

Check_Point = ModelCheckpoint(Log_Files_Path + "ep{epoch:02d}.h5",
                             monitor='val_loss', 
                             save_weights_only = True, 
                             save_best_only = True
                             )

Early_Stopping = EarlyStopping(monitor='val_loss', 
                               min_delta=0, 
                               patience=Patience, 
                               verbose=1, 
                               mode='auto')

CSV_Logger = CSVLogger(Log_Files_Path + "Logs.csv")

Call_Backs = [Tensor_Board, Check_Point, Early_Stopping, Visualizer, CSV_Logger]

GBM_WSSM_Results = my_model.fit_generator(Train_Augmentation, 
                                          len(X_Train), 
                                          Epochs, 
                                          verbose=1,
                                          validation_data=Test_Augmentation, 
                                          validation_steps=len(X_Validation),
                                          callbacks=Call_Backs
                                          )

##############################################################################

# Save GBM_WSSM Weights on Disk

Model_Weights_Path = os.path.join(Model_Outputs_Path, "GBM_WSSM.h5")
my_model.save_weights(Model_Weights_Path)




























