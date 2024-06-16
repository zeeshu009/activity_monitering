#!/usr/bin/env python
# coding: utf-8

# In[52]:

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from scipy.linalg import norm
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import io
import base64
import time
import seaborn as sns
import matplotlib.pyplot as plt



# ## Data Preprocessing

# In[53]:


tr_msAcc = np.load("dataset/bbh/training/trainMSAccelerometer.npy")
tr_msGyr = np.load("dataset/bbh/training/trainMSGyroscope.npy")
tr_labels = np.load("dataset/bbh/training/trainLabels.npy")

OPEN_DOOR = 20
RUB_HANDS = 36
DRINK = 10

tr_labels_OPEN_DOOR_idx = tr_labels == OPEN_DOOR
tr_labels_RUB_HANDS_idx = tr_labels == RUB_HANDS
tr_labels_DRINK_idx = tr_labels == DRINK

tr_msAcc_OPEN_DOOR = tr_msAcc[tr_labels_OPEN_DOOR_idx]
tr_msGyr_OPEN_DOOR = tr_msGyr[tr_labels_OPEN_DOOR_idx]

tr_msAcc_RUB_HANDS = tr_msAcc[tr_labels_RUB_HANDS_idx]
tr_msGyr_RUB_HANDS = tr_msGyr[tr_labels_RUB_HANDS_idx]

tr_msAcc_DRINK = tr_msAcc[tr_labels_DRINK_idx]
tr_msGyr_DRINK = tr_msGyr[tr_labels_DRINK_idx]

tr_labels_OPEN_DOOR = tr_labels[tr_labels_OPEN_DOOR_idx]
tr_labels_RUB_HANDS = tr_labels[tr_labels_RUB_HANDS_idx]
tr_labels_DRINK = tr_labels[tr_labels_DRINK_idx]

tr_msAcc_Three_Activities = np.concatenate((tr_msAcc_OPEN_DOOR, tr_msAcc_RUB_HANDS, tr_msAcc_DRINK))
tr_msGyr_Three_Activities = np.concatenate((tr_msGyr_OPEN_DOOR, tr_msGyr_RUB_HANDS, tr_msGyr_DRINK))
tr_labels_Three_Activities = np.concatenate((tr_labels_OPEN_DOOR, tr_labels_RUB_HANDS, tr_labels_DRINK))

np.save("train_MSAccelerometer_OpenDoor_RubHands.npy", tr_msAcc_Three_Activities)
np.save("train_MSGyroscope_OpenDoor_RubHands.npy", tr_msGyr_Three_Activities)
np.save("train_labels_OpenDoor_RubHands.npy", tr_labels_Three_Activities)


# In[54]:


print(tr_msAcc_Three_Activities.shape)
print(tr_msGyr_Three_Activities.shape)
print(tr_labels_Three_Activities.shape)


# In[55]:


ts_msAcc = np.load("dataset/bbh/testing/testMSAccelerometer.npy")
ts_msGyr = np.load("dataset/bbh/testing/testMSGyroscope.npy")
ts_labels = np.load("dataset/bbh/testing/testLabels.npy")

ts_labels_OPEN_DOOR_idx = ts_labels == OPEN_DOOR
ts_labels_RUB_HANDS_idx = ts_labels == RUB_HANDS
ts_labels_DRINK_idx = ts_labels == DRINK

ts_msAcc_OPEN_DOOR = ts_msAcc[ts_labels_OPEN_DOOR_idx]
ts_msGyr_OPEN_DOOR = ts_msGyr[ts_labels_OPEN_DOOR_idx]

ts_msAcc_RUB_HANDS = ts_msAcc[ts_labels_RUB_HANDS_idx]
ts_msGyr_RUB_HANDS = ts_msGyr[ts_labels_RUB_HANDS_idx]

ts_msAcc_DRINK = ts_msAcc[ts_labels_DRINK_idx]
ts_msGyr_DRINK = ts_msGyr[ts_labels_DRINK_idx]

ts_labels_OPEN_DOOR = ts_labels[ts_labels_OPEN_DOOR_idx]
ts_labels_RUB_HANDS = ts_labels[ts_labels_RUB_HANDS_idx]
ts_labels_DRINK = ts_labels[ts_labels_DRINK_idx]

ts_msAcc_Three_Activities = np.concatenate((ts_msAcc_OPEN_DOOR, ts_msAcc_RUB_HANDS,ts_msAcc_DRINK))
ts_msGyr_Three_Activities = np.concatenate((ts_msGyr_OPEN_DOOR, ts_msGyr_RUB_HANDS,ts_msGyr_DRINK))
ts_labels_Three_Activities = np.concatenate((ts_labels_OPEN_DOOR, ts_labels_RUB_HANDS,ts_labels_DRINK))

np.save("test_MSAccelerometer_OpenDoor_RubHands.npy", ts_msAcc_Three_Activities)
np.save("test_MSGyroscope_OpenDoor_RubHands.npy", ts_msGyr_Three_Activities)
np.save("test_labels_OpenDoor_RubHands.npy", ts_labels_Three_Activities)


# In[56]:


print(ts_msAcc_Three_Activities.shape)
print(ts_msGyr_Three_Activities.shape)
print(ts_labels_Three_Activities.shape)


# In[57]:


def compute_features(data,tr_msAcc_Three_Activities):
    for i in range(tr_msAcc_Three_Activities.shape[0]):
        # Initialize an empty list to hold statistics for this sample
        stats = []
    
        # Maximum
        stats.append(np.max(tr_msAcc_Three_Activities[i], axis = 0))
    
        # Minimum
        stats.append(np.min(tr_msAcc_Three_Activities[i], axis = 0))
    
        # First-order mean
        mean_val = np.mean(tr_msAcc_Three_Activities[i], axis = 0)
        stats.append(mean_val)
    
        # Standard Deviation
        stats.append(np.std(tr_msAcc_Three_Activities[i], axis = 0))
    
        # Percentile 50
        stats.append(np.percentile(tr_msAcc_Three_Activities[i], 50, axis = 0))
    
        # Percentile 80
        stats.append(np.percentile(tr_msAcc_Three_Activities[i], 80, axis = 0))
    
        # Norm of the first-order mean
        stats.append(np.full(mean_val.shape, norm(mean_val)))
    
        # Average (same as mean)
        stats.append(mean_val)
    
        # Interquartile range
        stats.append(np.percentile(tr_msAcc_Three_Activities[i], 75, axis = 0) - np.percentile(tr_msAcc_Three_Activities[i], 25, axis = 0))
    
        # Second-order mean
        squared_mean = np.mean(np.square(tr_msAcc_Three_Activities[i]), axis = 0)
        stats.append(squared_mean)
    
        # Skewness
        stats.append(skew(tr_msAcc_Three_Activities[i], axis = 0))
    
        # Norm of the second-order mean
        stats.append(np.full(squared_mean.shape, norm(squared_mean)))
    
        # Zero-crossing
        zero_crossings = np.sum(np.diff(np.sign(tr_msAcc_Three_Activities[i]), axis = 0) != 0, axis = 0)
        stats.append(zero_crossings)
    
        # Kurtosis
        stats.append(kurtosis(tr_msAcc_Three_Activities[i], axis = 0))
    
        # Spectral energy
        frequencies, power_spectral_density = welch(tr_msAcc_Three_Activities[i], axis = 0)
        spectral_energy = np.sum(power_spectral_density, axis = 0)
        stats.append(spectral_energy)
    
        # Percentile 20
        stats.append(np.percentile(tr_msAcc_Three_Activities[i], 20, axis = 0))
    
        # Auto-correlation (assuming lag 1)
        autocorr = np.array([acf(tr_msAcc_Three_Activities[i][:, j], nlags = 1, fft = True)[1] for j in range(tr_msAcc_Three_Activities[i].shape[1])])
        stats.append(autocorr)
    
        # Spectral entropy
        power_spectral_density /= np.sum(power_spectral_density, axis = 0, keepdims = True)
        spectral_entropy = entropy(power_spectral_density, axis = 0)
        stats.append(spectral_entropy)
    
        # Convert list of arrays to a 2D array of shape (18, 3)
        stats_array = np.array(stats)
    
        # Store in pre-allocated data array
        data[i] = stats_array
    
    # Now `data` contains the computed statistics for each sample


# # Training Data
# 

# In[58]:


data = np.empty((tr_msAcc_Three_Activities.shape[0], 18, 3))
compute_features(data,tr_msAcc_Three_Activities)
# reshape the data so that each row contain all features of the one example(x-axis,y-axis,z-axis)
data = np.reshape(data,(tr_msAcc_Three_Activities.shape[0],1,-1))
print(data.shape)
tr_msAcc_Three_Activities = data


# In[59]:


data = np.empty((tr_msGyr_Three_Activities.shape[0], 18, 3))
compute_features(data,tr_msGyr_Three_Activities)
data = np.reshape(data,(tr_msGyr_Three_Activities.shape[0],1,-1))
print(data.shape)
tr_msGyr_Three_Activities = data


# In[60]:


train_data = np.concatenate((tr_msAcc_Three_Activities, tr_msGyr_Three_Activities), axis=2)
train_labels = tr_labels_Three_Activities


train_data = np.squeeze(train_data, axis=1)
train_labels = train_labels[:, np.newaxis]


# CLASS IMBALANCE SOLUTION (OVERSAMPLING USING SMOTE)
smote = SMOTE(random_state=42)
train_data, train_labels = smote.fit_resample(train_data, train_labels.ravel())

# Standardize the data
Scaler = StandardScaler()
train_data = Scaler.fit_transform(train_data)

# train_data = preprocessing.normalize(train_data)

# Original labels    new lables
# OPEN_DOOR = 20 --> 0
# RUB_HANDS = 36 --> 1
# DRINK = 10 --> 2

count_20 = 0
count_36 = 0
count_9 = 0

for i in range(train_data.shape[0]):
    if train_labels[i] == OPEN_DOOR:
        train_labels[i] = 0
        count_20+=1
    
    elif train_labels[i] == RUB_HANDS:
        train_labels[i] = 1
        count_36+=1

    elif train_labels[i] == DRINK:
        train_labels[i] = 2
        count_9+=1

print('tr_count_20',count_20)
print('tr_count_36',count_36)
print('tr_count_9',count_9)

indices = np.random.permutation(train_data.shape[0])
train_data = train_data[indices]
train_labels = train_labels[indices]



# # Test Data

# In[61]:


data = np.empty((ts_msAcc_Three_Activities.shape[0], 18, 3))
compute_features(data,ts_msAcc_Three_Activities)
# reshape the data so that each row contain all features of the one example(x-axis,y-axis,z-axis)
data = np.reshape(data,(ts_msAcc_Three_Activities.shape[0],1,-1))
data[0,0,:]
print(data.shape)
ts_msAcc_Three_Activities = data


# In[62]:


data = np.empty((ts_msGyr_Three_Activities.shape[0], 18, 3))
compute_features(data,ts_msGyr_Three_Activities)
data = np.reshape(data,(ts_msGyr_Three_Activities.shape[0],1,-1))
data[0,0,:]
print(data.shape)
ts_msGyr_Three_Activities = data


# In[63]:


test_data = np.concatenate((ts_msAcc_Three_Activities, ts_msGyr_Three_Activities), axis = 2)
test_labels = ts_labels_Three_Activities


test_data = np.squeeze(test_data, axis = 1)
test_labels = test_labels[:, np.newaxis]

# test_data = preprocessing.normalize(test_data)


smote = SMOTE(random_state=42)
test_data, test_labels = smote.fit_resample(test_data, test_labels.ravel())


test_data = Scaler.fit_transform(test_data)

# Original labels  ->  new lables
# OPEN_DOOR = 20 --> 0
# RUB_HANDS = 36 --> 1
# DRINK = 4 --> 2

count_20 = 0
count_36 = 0
count_9 = 0
for i in range(test_data.shape[0]):
    if test_labels[i] == OPEN_DOOR:
        test_labels[i] = 0
        count_20 += 1

    elif test_labels[i] == RUB_HANDS:
        test_labels[i] = 1
        count_36 += 1

    elif test_labels[i] == DRINK:
        test_labels[i] = 2
        count_9 += 1

print('ts_count_20', count_20)
print('ts_count_36', count_36)
print('ts_count_9', count_9)

indices = np.random.permutation(test_data.shape[0])
test_data = test_data[indices]
test_labels = test_labels[indices]

# # RandomForestClassifier

# In[64]:


# Initialize, train, and make predictions with the Random Forest classifier
def rf_classifier():
    # Initialize, train, and make predictions with the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_data, train_labels)
    y_pred = rf_classifier.predict(test_data)

    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='weighted')
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot evaluation metrics and confusion matrix
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    # Bar plot for accuracy
    ax[0].bar(['Accuracy'], [accuracy], color='blue')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Score')
    ax[0].set_title('Accuracy')
    
    # Bar plot for F1 score
    ax[1].bar(['F1 Score'], [f1], color='green')
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel('Score')
    ax[1].set_title('F1 Score')
    
    # Heatmap for confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[2],
                xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    ax[2].set_xlabel('Predicted Labels')
    ax[2].set_ylabel('True Labels')
    ax[2].set_title('Confusion Matrix')
    
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return accuracy,f1,conf_matrix,graph_url


# # Support Vector Machine

# In[71]:


# Initialize, train, and make predictions with the SVM classifier
def svm_classifier():
    # Initialize, train, and make predictions with the SVM classifier
    svm_classifier = SVC(kernel='rbf',random_state=42)
    svm_classifier.fit(train_data, train_labels)
    y_pred = svm_classifier.predict(test_data)
    
    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='weighted')
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot evaluation metrics and confusion matrix
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    # Bar plot for accuracy
    ax[0].bar(['Accuracy'], [accuracy], color='blue')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Score')
    ax[0].set_title('Accuracy')
    
    # Bar plot for F1 score
    ax[1].bar(['F1 Score'], [f1], color='green')
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel('Score')
    ax[1].set_title('F1 Score')
    
    # Heatmap for confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[2],
                xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    ax[2].set_xlabel('Predicted Labels')
    ax[2].set_ylabel('True Labels')
    ax[2].set_title('Confusion Matrix')
    
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return accuracy,f1,conf_matrix,graph_url
# In[ ]:




