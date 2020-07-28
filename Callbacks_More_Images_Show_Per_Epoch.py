# Import Required Modules

import os
import json
import random
import numpy as np
from skimage.io import imshow 
import matplotlib.pyplot as plt
import keras.callbacks as callbacks
from Mask_Integer_Encoding import RGBs_to_Integers

##############################################################################

hight, width, channel, patch_size = 224, 224, 3, 224
input_shape = (hight, width, channel)



def color_label(img, id2code):
    rows, cols = img.shape
    result = np.zeros((rows, cols, 3), 'uint8')
    for j in range(rows):
        for k in range(cols):
            result[j, k] = id2code[img[j, k]]
    return result


rand_height = random.randint(0, hight - patch_size)
rand_width = random.randint(0, width - patch_size)


class WeightVisualizerCallback(callbacks.Callback):
#    def __init__(self, images, masks, figsize=None, mask_decoding_func=None):
    def __init__(self, images, masks, figPath, figPath2, jsonPath, H, W, patch_size, figsize, startAt =0 ):

        super(WeightVisualizerCallback, self).__init__()
        assert images.shape[:-1] == masks.shape[:-1]
        if isinstance(figsize, int):
            figsize = (figsize, figsize)
        if figsize is None:
            figsize = tuple(images.shape[1:3])
#        if mask_decoding_func is None:
#            mask_decoding_func = identity
        self.figsize = figsize
        self.images = images
        self.masks = masks
        self.figPath = figPath
        self.figPath2 = figPath2
        self.jsonPath = jsonPath
        self.startAt = startAt
        
        self.rand_height = random.randint(0, H - patch_size)
        self.rand_width = random.randint(0, W - patch_size)
        self.patch_size = patch_size
        
        self.image_patch = images[0, 0:self.patch_size, 0:self.patch_size, :]
        self.mask_patch = masks[0, 0:self.patch_size, 0:self.patch_size, :]
        
        print(self.image_patch.shape)
        print(self.mask_patch.shape)
        
    def on_train_begin(self, logs={}):
        self.H = {}
        
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
        

    def on_epoch_end(self, epoch, logs={}):

        predicted_masks_1 = self.model.predict(np.expand_dims(self.image_patch, 0), 1)
        
        input_shape = (hight, width, channel)

        predicted_masks = np.argmax(predicted_masks_1, axis=-1)                    # int64   (1, 65536)
        outcome = np.resize(predicted_masks, (input_shape[0], input_shape[1]))     # unit8   (256, 256)
        Mask_RGBs, Mask_Regions, RGBs_to_Integer = RGBs_to_Integers('.../.../Mask_Labels.txt')

 

        print(list(zip(Mask_RGBs, Mask_Regions)))
        Integers_to_RGBs = {val: key for (key, val) in RGBs_to_Integer.items()}
        
        
        
        outcome = color_label(outcome, Integers_to_RGBs)     # unit8   (256, 256, 3)  --> Final Image
        
        
        print('Image')
        imshow(self.image_patch)
        plt.show()
        
        print('Mask')
        imshow(self.mask_patch)
        plt.show()
        
        print('Predicted_Mask')
        imshow(outcome)
        plt.show()
        
        for (k,v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
            
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            
            
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label = "Train_Loss")
            plt.plot(N, self.H["val_loss"], label = "Validation_Loss")
            plt.title("Training & Validation Loss  [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.savefig(self.figPath)
            plt.close()
            
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["acc"], label = "Train_Accuracy")
            plt.plot(N, self.H["val_acc"], label = "Validation_Accuracy")
            plt.title("Training & Validation Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            
            plt.savefig(self.figPath2)
            plt.close()
            

            
            
            
            
            
            

        
