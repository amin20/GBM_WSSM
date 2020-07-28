# Import Required Modules

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.layers import Input
from keras.models import Model
Image.MAX_IMAGE_PIXELS = 1000000000
from GBM_WSSM import GBM_WSSM_Model
from Mask_Integer_Encoding import RGBs_Finder, RGBs_to_Integers

##############################################################################

# Parse Arguments

def parse_args(arguments):
    
    GBM_Parser = argparse.ArgumentParser(description = "This part is being used to fine-tune GBM_WSSM on the test samples")
    
    # All the necessary arguments (variables) are defined as follow 
    
        
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

    return GBM_Parser.parse_args(arguments)

##############################################################################

# Import all Arguments

All_Arguments = sys.argv[1:]
All_Arguments = parse_args(All_Arguments)

GBM_Test_Images_Path = All_Arguments.GBM_Test_Images_Path
Predicted_Masks_Path = All_Arguments.Predicted_Masks_Path
GBM_WSSM_h5_path = All_Arguments.GBM_WSSM_h5_path
Labels_txt = All_Arguments.Labels_txt
Input_Shape = All_Arguments.Image_Info[0]
Class_No = All_Arguments.Image_Info[1]

##############################################################################

Test_Image_Path = next(os.walk(GBM_Test_Images_Path))[2]
Image_Input = Input(shape = Input_Shape)
Output = GBM_WSSM_Model(Class_No, Image_Input)
my_model = Model(inputs = Image_Input, outputs = Output)
my_model.load_weights(GBM_WSSM_h5_path)

sys.stdout.flush()
for i, file_name_ in tqdm(enumerate(Test_Image_Path), total = len(Test_Image_Path)):

    
    Image_Name = os.path.join(GBM_Test_Images_Path, file_name_)
    Image_Object = Image.open(Image_Name).resize((Input_Shape[0], Input_Shape[1]), Image.NEAREST)
    Image_Array = np.array(Image_Object)
    Normalized_Image = Image_Array / 255.

    Mask_Predicted = my_model.predict(np.expand_dims(Normalized_Image, 0), 1)
    Mask_Predicted = np.argmax(Mask_Predicted, axis=-1)

    Final_Mask_Size = np.resize(Mask_Predicted, (Input_Shape[0], Input_Shape[1]))
    RGB_Values_List, Labels_List, RGB_and_Integers_Dictionary = RGBs_to_Integers(Labels_txt)

    Integer_to_RGBs = {val: key for (key, val) in RGB_and_Integers_Dictionary.items()}
    Final_Mask = RGBs_Finder(Final_Mask_Size, Integer_to_RGBs)
    Mask = Image.fromarray(Final_Mask)    
    Mask.save(Predicted_Masks_Path + "/" + file_name_)
