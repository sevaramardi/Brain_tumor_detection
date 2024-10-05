import os
import scipy
import cv2
import h5py
import numpy as np
from PIL import Image
from mat73 import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


directory = 'dataset_path'
num = 0
for file in os.listdir(directory):
   num+=1
   if file.endswith('.mat'):
        path = os.path.join(directory,file)
        with h5py.File(path, 'r') as f:
            img = f['cjdata']['image']
            label = f['cjdata']['label'][0][0]
            tumorBorder = f['cjdata']['tumorBorder'][()]
            mask = f['cjdata']['tumorMask'][()]

            img = np.array(img, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)

            tumorBorder = tumorBorder.reshape(-1, 2)
            x_min = tumorBorder[:, 0].min()
            x_max = tumorBorder[:, 0].max()
            y_min = tumorBorder[:, 1].min()
            y_max = tumorBorder[:, 1].max()
            file_path = f'directory_{num}.txt'

           
            with open(file_path, 'w') as label_file:
                label_file.write(f'{int(label)} {x_min} {y_min} {x_max} {y_max}\n')
            plt.imsave(f'directory_{num}.jpg',img,cmap='gray')
            plt.imsave(f'directory_{num}.jpg',mask,cmap='gray')

print('Process is succesfully done!')



    
