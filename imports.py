from google.colab import drive
drive.mount('/content/drive')
!pip install SimpleITK
!pip install nipype
!pip install pyradiomics
!pip install h5py

# General libraries
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Image processing
from PIL import Image, ImageEnhance
import cv2
from skimage.io import imread, imshow
from skimage import exposure, transform, util, color
from skimage.filters import gaussian
from skimage.feature import hog
from scipy.signal import find_peaks

# Medical imaging
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.utils import shuffle

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing import image
