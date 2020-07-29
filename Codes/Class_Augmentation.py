# Import Required Modules

import numpy as np
import random as RD
from Class_Batch_Index_Generator import Batch_Index_Generator


################################################################################################################
        
class Augmentation(object):
    
    def __init__(self, 
                 Batch_Size, 
                 Images, 
                 Masks, 
                 Crop_Size, 
                 Train_Phase = True
                 ):
        
        self.Batch_Size = Batch_Size
        self.Images = Images
        self.Masks = Masks
        self.Crop_Size = Crop_Size
        self.Train_Phase = Train_Phase
        
        self.Masks_Channels = 1
        self.Images_No = Images.shape[0]
        self.Images_Height = Images.shape[1]        
        self.Images_Width = Images.shape[2]
        self.Model_Input_Rows, self.Model_Inputs_Columns = Crop_Size[0], Crop_Size[1]
        self.Index_Generator = Batch_Index_Generator(self.Images_No, Batch_Size, Train_Phase)       # Object from class Batch_Index_Generator --> 
                                                                                                    # Index_Generator --> Indices_List from Batch_Size
                                                                                                    
    def Slice_Finder(self,
                     Original_Input_Size,
                     Desired_Output_Size
                     ):
        
        if self.Train_Phase:
            Slice_Start_Point = RD.randint(0, Original_Input_Size - Desired_Output_Size)
        
        else: 
            Slice_Start_Point = Original_Input_Size - Desired_Output_Size
            
        Slice_Final = slice(Slice_Start_Point, Slice_Start_Point + Desired_Output_Size)
        
        return Slice_Final
    
    
    
    def Image_Finder(self,
                     Index
                     ):
        
        Rows_Slice = self.Slice_Finder(self.Images_Height, self.Model_Input_Rows) 
        Columns_Slice = self.Slice_Finder(self.Images_Width, self.Model_Inputs_Columns)
        
        Cropped_Image = self.Images[Index, Rows_Slice, Columns_Slice]                               # Cropped image from Current_Image
        cropped_Mask = self.Masks[Index, Rows_Slice, Columns_Slice]                                 # Cropped mask from Current_Image
        
        if self.Train_Phase and (RD.random()>0.5):
        
            Cropped_Image = Cropped_Image[:, ::-1]
            cropped_Mask = cropped_Mask[:, ::-1]
            
        
        return Cropped_Image, cropped_Mask
            
    
    def __next__(self,
                 ):
        
        Index_List = next(self.Index_Generator)
    
        Images = (self.Image_Finder(Index) for Index in Index_List)
        
        Cropped_Images, cropped_Masks = zip(*Images)
        
        return np.stack(Cropped_Images), np.stack(cropped_Masks).reshape(len(cropped_Masks), -1, self.Masks_Channels)

        

                
