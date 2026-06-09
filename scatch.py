import os
import numpy as np
import random

import matplotlib.pyplot as plt

data_path = './data_stft_64'
path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

arr_dict = dict()
label_dict = dict()

for folder_path in path_list:
    bearing_type = folder_path.split("/")[-1]
    
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.npz')]
    file_names = random.sample(file_names, 1)
    
    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        x = npz['x']
        y = npz['y']
        
        npy_list.append(x)
        label_list.append(y)
        npy_arr = np.vstack(npy_list)
        label_arr = np.hstack(label_list)
    
    arr_dict[bearing_type] = npy_arr
    label_dict[bearing_type] = label_arr

type = 'SB'
b = arr_dict[f'{type}'][1]

plt.imshow(b, aspect='auto', origin='lower')
plt.title('Spectrogram')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.savefig(f'./plot/Spectrogram-{type}')
plt.close()
