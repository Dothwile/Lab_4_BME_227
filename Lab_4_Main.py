'''
Lab_4_Main
Created by Artur Smiechowski 3/22/21
'''

# %% Imports
import numpy as np

# %% Part 1

# Load in the epoch data
emg_epoch = np.load('C:/Users/Artur Smiechowski/Documents/BME227_Code/Lab_3_BME_227/Data_Values_epochs.npy')

emg_epoch_var = np.transpose(np.var(emg_epoch, 0)) # 2 dimensional [channel, epoch]
# Is squeeze to catch incomplete epochs?

is_true_right = np.zeros(np.shape(emg_epoch)[2])
for epoch_n in  range(len(is_true_right)):
    # Rest 1 sec : Left 1 sec : Right 1 sec : Both 1 sec
    # Epoch ~200ms
    if (epoch_n % 