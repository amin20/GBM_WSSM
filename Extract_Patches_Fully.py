# Import Required Modules

import os
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from skimage.io import imread, imsave

##############################################################################

Prefered_Patch_Size = 512

Image_Path = ".../Images/"                             # Path to Image Folder
Image_Output_Path = ".../Images/"                      # Path to Image Folder to Save Extracted Patches
Mask_Path = ".../Masks/"                               # Path to Mask Folder
Mask_Output_Path = ".../Masks/"                        # Path to Mask Folder to Save Extracted Patches


Image_Size = sorted(next(os.walk(Image_Path))[2])
Mask_Size = sorted(next(os.walk(Mask_Path))[2])


def Patch_Extractor(Input_Path, Output_Path, Size):
    
    for index, file_name in tqdm(enumerate(Size), total = len(Size)):
        
        counter = 0
        image = imread(Input_Path + file_name)
        H, W, CH = image.shape[0], image.shape[1], image.shape[2]
        Total_Patches_in_rows = int(H / Prefered_Patch_Size)
        Total_Patches_in_Cols = int(W / Prefered_Patch_Size)
        name, ext = os.path.splitext(file_name)
        
        for i in range(0, Total_Patches_in_Cols):
            
            Step_Rows = Prefered_Patch_Size * i
            
            for j in range(0, Total_Patches_in_rows):    
              
              Step_Cols = Prefered_Patch_Size * j
              
              Extracted_Patch = image[Step_Rows : Step_Rows + Prefered_Patch_Size, Step_Cols: Step_Cols + Prefered_Patch_Size, :3]

              counter+=1
              
              file_name = '{file_name}{Separator}{Index1}{Format}'.format(file_name = name,  Separator = "_",  
                           Index1 = str(counter).zfill(2), Format = ".png")
              file_name = os.path.join(Output_Path, file_name)
              imsave(file_name, Extracted_Patch)
          

Patch_Extractor(Image_Path, Image_Output_Path, Image_Size)     
Patch_Extractor(Mask_Path, Mask_Output_Path, Mask_Size)
            
        