'''
Lab_4_Main
Created by Artur Smiechowski 3/22/21
'''

# %% Imports
import numpy as np

# %% Part 1

# Load in the epoch data
data_file_path = '.'
emg_epoch = np.load(data_file_path + 'Data_Values_epochs.npy')

emg_epoch_var = np.transpose(np.var(emg_epoch, 0)) # 2 dimensional [channel, epoch]
# Is squeeze to catch incomplete epochs?

# Create "Action Cycles" to simplify boolean array creation
# Rest 1 sec : Left 1 sec : Right 1 sec : Both 1 sec
# Epoch ~200ms
left_cycle = np.array([False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,True,True,True,])
right_cycle = np.array([False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,])

# Use np.tile to repeat the action cycles into expected value arrays
is_true_left = np.tile(left_cycle, np.shape(emg_epoch)[2]/20) # np.shape(emg_epoch)[2]/20 is numder of "Action Cycles" in a sample set
is_true_right = np.tile(right_cycle, np.shape(emg_epoch)[2]/20)

