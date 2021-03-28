'''
Lab_4_Main
Created by Artur Smiechowski 3/22/21
'''

# %% Imports
import numpy as np
from matplotlib import pyplot as plt

# %% Part 1

# Load in the epoch data //here lab 3 data is placed in lab4 folder for ease of access
data_file_path = 'C:/Users/Artur Smiechowski/Documents/BME227_Code/Lab_4_BME_227/Data_Values_epochs.npy'
emg_epoch = np.load(data_file_path)

emg_epoch_var = np.transpose(np.var(emg_epoch, 0)) # 2 dimensional [channel, epoch]
# Is squeeze to catch incomplete epochs?

# Create "Action Cycles" to simplify boolean array creation
# Rest 1 sec : Left 1 sec : Right 1 sec : Both 1 sec
# Epoch ~200ms
left_cycle = np.array([False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,True,True,True,])
right_cycle = np.array([False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,])

# Use np.tile to repeat the action cycles into expected value arrays (+1 to account for non-even division, then indexed into final array to remove extra values)
is_true_left = np.array(np.tile(left_cycle, int(np.shape(emg_epoch)[2]/20)+1)[:np.shape(emg_epoch)[2]]) # np.shape(emg_epoch)[2]/20 is numder of "Action Cycles" in a sample set
is_true_right = np.array(np.tile(right_cycle, int(np.shape(emg_epoch)[2]/20)+1)[:np.shape(emg_epoch)[2]]) 

# %% Part 2

plt.figure()

plt.xlabel("Variance on Channel 0 (V^2)")
plt.ylabel("Number of Trials")
plt.title("HMI Histogram for Left Fist Squeeze")
plt.hist(emg_epoch_var[is_true_left,0],bins=8, alpha=0.5) # Multidimensional indexing is seemingly wonky? [x][y] does not work intermittently
plt.hist(emg_epoch_var[~is_true_left,0],bins=8, alpha=0.5)