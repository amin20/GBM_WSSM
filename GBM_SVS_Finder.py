# Import Required Modules

import os
import sys
import tqdm
import openpyxl as xl
from shutil import copyfile

##############################################################################

Image_Dataset = '/.../Dataset_GDC/'                           # TCGA SVS Images path
Annotation_Path = '/.../'                                     # TCGA Log File path
Saved_Path = "/.../T1_TCGA_GBMs/"                             # GBM SVS File path to Save


# Import Dataset

Image_Folder = next(os.walk(Image_Dataset))[1]
workbook = xl.load_workbook(filename = Annotation_Path + "GBM_Logs.xlsx")
ws = workbook.active

# Find SVS Files

counter=1

sys.stdout.flush()
for i, folder_name in tqdm.tqdm(enumerate(Image_Folder), total = len(Image_Folder)):
    
    
    Image_SVS_Path = Image_Dataset + folder_name
    Image_SVS = next(os.walk(Image_SVS_Path))[2]
#    Search_String = Image_SVS[0]
    for j in Image_SVS:
        Base_Name = os.path.basename(j)
        Patient_ID, Extension = os.path.splitext(Base_Name)
        
        Total_Patients = 0
        
        
        for Columns in ws.rows:                               # Move throgh the rows 
    
            Partail_ID = Columns[1].value
            
            if Partail_ID in Patient_ID:
                           
                if (str(Columns[0].value == 'Glioblastoma')):
                    if not os.path.exists(Saved_Path + Partail_ID):
                        os.mkdir(Saved_Path + Partail_ID)
                    Image_Folder_New = Saved_Path + Partail_ID 
                    copyfile(Image_SVS_Path + '/' + j, Image_Folder_New + '/' + j)
                        
                    counter+=1
        