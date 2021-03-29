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

# Calculate variance and drop into emg_epoch_var, using nanvar to ignore dropped values
emg_epoch_var = np.transpose(np.nanvar(emg_epoch,axis=0)) # 2 dimensional [channel, epoch]

# Create "Action Cycles" to simplify boolean array creation
# Rest 1 sec : Left 1 sec : Right 1 sec : Both 1 sec
# Epoch ~200ms
left_cycle = np.array([False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,True,True,True,])
right_cycle = np.array([False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,])

# Use np.tile to repeat the action cycles into expected value arrays (+1 to account for non-even division, then indexed into final array to remove extra values)
is_true_left = np.array(np.tile(left_cycle, int(np.shape(emg_epoch)[2]/20)+1)[:np.shape(emg_epoch)[2]]) # np.shape(emg_epoch)[2]/20 is numder of "Action Cycles" in a sample set
is_true_right = np.array(np.tile(right_cycle, int(np.shape(emg_epoch)[2]/20)+1)[:np.shape(emg_epoch)[2]]) 

# %% Part 2

# Left Hand ---------------------------------------------------------------------------------------------

# Clear figure to be safe
plt.clf()

# Create figure to plot to
plt.figure()

# Create plot labels
plt.xlabel("Variance on Channel 0 (V^2)")
plt.ylabel("Number of Trials")
plt.title("HMI Histogram for Left Fist Squeeze")

# Create a Criterion point to divide data classification
criterion_left = 0.003 # Even after several new trials data improved but was not ultimately good (my arms really hurt from changing pad locations several times)

# Plot variances during expected true vs expected false epochs
plt.hist(emg_epoch_var[is_true_left,0],bins=8, alpha=0.5, label="True Left") # Multidimensional indexing is seemingly wonky? [x][y] does not work intermittently
plt.hist(emg_epoch_var[~is_true_left,0],bins=8, alpha=0.5, label="Not True Left")
plt.axvline(x=criterion_left, label="Threshhold") # Creates a vertical line at the criterion
plt.legend() # Enables data legend

# Save the figure
plt.savefig("C:/Users/Artur Smiechowski/Documents/BME227_Code/Lab_4_BME_227/HMI_Histogram_Left.png") # Permission errors thrown if not full path

is_predicted_left = emg_epoch_var[:,0] > criterion_left

# Right Hand ---------------------------------------------------------------------------------------------

# Clear figure to be safe
plt.clf()

# Create figure to plot to
plt.figure()

# Create plot labels
plt.xlabel("Variance on Channel 1 (V^2)")
plt.ylabel("Number of Trials")
plt.title("HMI Histogram for Right Fist Squeeze")

# Create a Criterion point to divide data classification
criterion_right = 0.0007 # Even after several new trials data improved but was not ultimately good (my arms really hurt from changing pad locations several times)

# Plot variances during expected true vs expected false epochs
plt.hist(emg_epoch_var[is_true_right,1],bins=8, alpha=0.5, label="True Right") # Multidimensional indexing is seemingly wonky? [x][y] does not work intermittently
plt.hist(emg_epoch_var[~is_true_right,1],bins=8, alpha=0.5, label="Not True Right")
plt.axvline(x=criterion_right, label="Threshhold") # Creates a vertical line at the criterion
plt.legend() # Enables data legend

# Save the figure
plt.savefig("C:/Users/Artur Smiechowski/Documents/BME227_Code/Lab_4_BME_227/HMI_Histogram_Right.png") # Permission errors thrown if not full path

is_predicted_right = emg_epoch_var[:,1] > criterion_right


''' Is criterion same for left and right?
No, the data is already rather messy so it shouldn't be expected that the signal curves intersect at the same point
This could be due to different placement on each respective arm, slight differences in the patients' left and right arms, as well as the innaccuracy of timing when squeezing your fists
'''

# %% Part 3

