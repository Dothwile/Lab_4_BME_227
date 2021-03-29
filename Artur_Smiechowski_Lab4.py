'''
Lab_4_Main
Created by Artur Smiechowski 3/22/21
A program that
    Loads up epoched emg data and compares it to expected "true" data
    Creates a criterion point for the left and right channels each to predict an action from the data
    Graphs the true vs not true predictions of the data for both channels
    Calculates a number of parameters for the data set (ITR, accuracy, etc)
    Reclassifies 2 channel 2 action sets into 4 actions
    Plots the total confusion matrix for this 4 action array
    Calculates the total accuracy and ITRs for the 4 action array
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
criterion_left = 0.0025 # Even after several new trials data improved but was not ultimately good (my arms really hurt from changing pad locations several times)

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

# Left Hand ---------------------------------------------------------------------------------------------

# Initialize true and false positives/negatives
true_positive_left = 0
false_positive_left = 0
true_negative_left = 0
false_negative_left = 0

# Loop through the data and sum true/false positive/negative
for epoch_n in range(len(emg_epoch_var[:])):
    # Check if predicted true matches expected true
    if is_predicted_left[epoch_n]:
        if is_predicted_left[epoch_n] == is_true_left[epoch_n]:
            true_positive_left = true_positive_left + 1
        else:
            false_positive_left = false_positive_left + 1
    # Check if predicted false matches expected negative
    else:
        if is_predicted_left[epoch_n] == is_true_left[epoch_n]:
            true_negative_left = true_negative_left + 1
        else:
            false_negative_left = false_negative_left + 1
            
# Calculate accuracy of channel
accuracy_left = (true_positive_left + true_negative_left) / len(emg_epoch_var[:])

# Sensitivty = True Pos / (True Pos + False Neg)
sensitivity_left = true_positive_left/(true_positive_left+false_negative_left)

# Specificity = True Neg / (True Neg + False Pos)
specificity_left = true_negative_left/(true_negative_left+false_positive_left)

# Information transfer rate per second
ITRs_left = 5*(np.log2(2)+accuracy_left*np.log2(accuracy_left)+(1-accuracy_left)*np.log2((1-accuracy_left)/(2-1))) # Find ITR per trial, ITRs is that *5 since 5 epochs per second

# Right Hand ---------------------------------------------------------------------------------------------

# Initialize true and false positives/negatives
true_positive_right = 0
false_positive_right = 0
true_negative_right = 0
false_negative_right = 0

# Loop through the data and sum true/false positive/negative
for epoch_n in range(len(emg_epoch_var[:])):
    # Check if predicted true matches expected true
    if is_predicted_right[epoch_n]:
        if is_predicted_right[epoch_n] == is_true_right[epoch_n]:
            true_positive_right = true_positive_right + 1
        else:
            false_positive_right = false_positive_right + 1
    # Check if predicted false matches expected negative
    else:
        if is_predicted_right[epoch_n] == is_true_right[epoch_n]:
            true_negative_right = true_negative_right + 1
        else:
            false_negative_right = false_negative_right + 1
            
# Calculate accuracy of channel
accuracy_right = (true_positive_right + true_negative_right) / len(emg_epoch_var[:])

# Sensitivty = True Pos / (True Pos + False Neg)
sensitivity_right = true_positive_right/(true_positive_right+false_negative_right)

# Specificity = True Neg / (True Neg + False Pos)
specificity_right = true_negative_right/(true_negative_right+false_positive_right)

# Information transfer rate per second
ITRs_right = 5*(np.log2(2)+accuracy_right*np.log2(accuracy_right)+(1-accuracy_right)*np.log2((1-accuracy_right)/(2-1))) # Find ITR per trial, ITRs is that *5 since 5 epochs per second

# %% Part 4

# Create the new prediction arrays, intially will be copies of other choice arrays but will update
# This implies using the same criterions as before, could implement different ones by using same process as before with diff number
is_predicted_4_left = np.copy(is_predicted_left) # TIL Python passes by reference, just = wont copy, just gives another name
is_predicted_4_right = np.copy(is_predicted_right)
is_predicted_4_click = np.copy(is_predicted_left)

# Update the prediction arrays
for epoch_n in range(len(is_predicted_4_left)): # I should standardize what values I reference for similar length applications, is there a best practice for this class?
    # Piggyback off the loop to update 4_click
    is_predicted_4_click[epoch_n] = False
    
    if is_predicted_4_left[epoch_n]: # Check if left is high
        if is_predicted_4_right[epoch_n]: # If Right also high Click
            # Update the truth arrays
            is_predicted_4_click[epoch_n] = True
            is_predicted_4_left[epoch_n] = False
            is_predicted_4_right[epoch_n] = False
             
'''     
# Note I started writing the code before finishing reading the section
# I thought we where actually implementing an HMI
# These sections of the if/else are not necessary as the array is already correct at epoch_n to get here
# You only actually need the 2 checks and one update set to properly create 4 options
# Nvm you could just use an and
# Leaving this in comment to help clarify above
# There is a time for refactoring but I have an exam to study for
   
   else: # Move left
            pass      
    else: # When left is low
        if is_predicted_4_right[epoch_n]: # Move right
            pass      
        else: # Rest
            pass
'''

# Create new truth arrays
is_true_4_left = np.copy(is_true_left)
is_true_4_right = np.copy(is_true_right)
is_true_4_click = np.copy(is_true_left)

# Update the truth arrays, similar to prediction
for epoch_n in range(len(is_true_4_left)): # I should standardize what values I reference for similar length applications, is there a best practice for this class?
    # Piggyback off the loop to update 4_click
    is_true_4_click[epoch_n] = False
    
    if is_true_4_left[epoch_n]: # Check if left is high
        if is_true_4_right[epoch_n]: # If Right also high Click
            # Update the truth arrays
            is_true_4_click[epoch_n] = True
            is_true_4_left[epoch_n] = False
            is_true_4_right[epoch_n] = False


# Create an action list
actions = ['left','right','click','rest']

# Create a 2D confusion matrix (left,right,click,rest) in that order both dimensions
# Rows are predicted, columns are actual
confusion_matrix = np.zeros((4,4))

# Update the array with the true vs predicted actions; Done column-wise sweep
# True Left
confusion_matrix[0,0] = np.sum((is_true_4_left==True)&(is_predicted_4_left==True)) # Don't think ==True is necessary of efficient since the arrays are already Bools(~ or not to negate), but will keep for clarity
confusion_matrix[1,0] = np.sum((is_true_4_left==True)&(is_predicted_4_right==True))
confusion_matrix[2,0] = np.sum((is_true_4_left==True)&(is_predicted_4_click==True))
confusion_matrix[3,0] = np.sum((is_true_4_left==True)&((is_predicted_4_click==False)&(is_predicted_4_left==False)&(is_predicted_4_right==False))) # The long triple check since when 4_click is set it falisifes left/right, could get clash
# True Right
confusion_matrix[0,1] = np.sum((is_true_4_right==True)&(is_predicted_4_left==True))
confusion_matrix[1,1] = np.sum((is_true_4_right==True)&(is_predicted_4_right==True))
confusion_matrix[2,1] = np.sum((is_true_4_right==True)&(is_predicted_4_click==True))
confusion_matrix[3,1] = np.sum((is_true_4_right==True)&((is_predicted_4_click==False)&(is_predicted_4_left==False)&(is_predicted_4_right==False)))
# True Click
confusion_matrix[0,2] = np.sum((is_true_4_click==True)&(is_predicted_4_left==True))
confusion_matrix[1,2] = np.sum((is_true_4_click==True)&(is_predicted_4_right==True))
confusion_matrix[2,2] = np.sum((is_true_4_click==True)&(is_predicted_4_click==True))
confusion_matrix[3,2] = np.sum((is_true_4_click==True)&((is_predicted_4_click==False)&(is_predicted_4_left==False)&(is_predicted_4_right==False)))
# True Rest
confusion_matrix[0,3] = np.sum(((is_true_4_right==False)&(is_true_4_left==False)) &(is_predicted_4_left==True)) # There should have been a true_rest array
confusion_matrix[1,3] = np.sum(((is_true_4_right==False)&(is_true_4_left==False)) &(is_predicted_4_right==True))
confusion_matrix[2,3] = np.sum(((is_true_4_right==False)&(is_true_4_left==False)) &(is_predicted_4_click==True))
confusion_matrix[3,3] = np.sum(((is_true_4_right==False)&(is_true_4_left==False)) &((is_predicted_4_click==False)&(is_predicted_4_left==False)&(is_predicted_4_right==False)))

# Prepare the figure to plot the confusion matrix
plt.clf()

# Figure labels
plt.title("HMI Confusion Matrix")
plt.ylabel("Predicted Action")
plt.xlabel("Actual Acion")
# Action labels
plt.xticks([0,1,2,3],actions)
plt.yticks([3,2,1,0],actions)

# Plot the matrix
plt.pcolor(np.flip(confusion_matrix,0)) # Flipped vertically to match stated oreinatation/variable explorer orientation

# Add a color bar
plt.colorbar(label="Number of Trials")

# Save the confusion matrix
plt.savefig("C:/Users/Artur Smiechowski/Documents/BME227_Code/Lab_4_BME_227/HMI_Confusion_Matrix.png")

# Calculate Total Accuracy
accuracy_total = np.sum(confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2]+confusion_matrix[3,3])/np.sum(confusion_matrix[:,:])
# Calculate Information Transfer Rate Total
ITRs_total = 5*np.log2(4)+accuracy_total*np.log2(accuracy_total)+(1-accuracy_total)*np.log2((1-accuracy_total)/(4-1)) # *5 since 5 epochs per second