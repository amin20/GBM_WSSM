# Import Required Madules

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

###########################################################################################


def One_Image_or_GT_Loading(File_Path, 
                            New_Size, 
                            Size_Boolean
                            ):
    
    if Size_Boolean:
        
        Image_Object = Image.open(File_Path).resize(New_Size, Image.NEAREST)
    
    Image_Object = Image.open(File_Path)
    Image_Final = np.array(Image_Object)
    return Image_Final

def Images_or_GTs_Path(Images_Path, 
                       New_Size,
                       Size_Boolean
                       ):
    
    All_Images_Path = sorted(glob(Images_Path + "*.png"))
    
    sys.stdout.flush() 
    All_Images_Array = np.stack([One_Image_or_GT_Loading(One_Image_Path, New_Size, Size_Boolean) for One_Image_Path in tqdm(All_Images_Path, total = len(os.listdir(Images_Path)))])
        
    return All_Images_Array

def Normalize(
        Images_array
        ):
    
    Images_Normalize = np.zeros((Images_array.shape), dtype = np.float32)
    for image_index in tqdm(range(len(Images_Normalize)), total = len(Images_Normalize)):
        
        Images_Normalize[image_index] = Images_array[image_index] / 255.
        
    return Images_Normalize
        