# Import Required Modules

import threading
import numpy as np

################################################################################################################


class Batch_Index_Generator(object):
    
    def __init__(self,
                 Image_No, 
                 Batch_Size,
                 Train_Phase = False
                 ):
        
        self.Image_No = Image_No
        self.Batch_Size = Batch_Size
        self.Train_Phase = Train_Phase                  # If we are in testing phase, the samples indices should not be permutated! 
        
        self.Resources_Lock = threading.Lock()
        self.Index_Reset()
        
    def Index_Reset(self
                        ):
        
        if self.Train_Phase:
            self.Index_List = np.random.permutation(self.Image_No)                                   # Permutated List
        
        else:
            self.Index_List = np.arange(0, self.Image_No)                                            # Sorted List
        
        self.index_counter = 0
        
    def __next__(self
                 ):
        
        with self.Resources_Lock:
                
            if self.index_counter >= self.Image_No:
                self.Index_Reset()
            
            Index_No = min(self.Batch_Size, self.Image_No - self.index_counter)                      # Batch_Size
            Index_List_Final = self.Index_List[self.index_counter:self.index_counter + Index_No] 
            self.index_counter += Index_No
            
            return Index_List_Final
