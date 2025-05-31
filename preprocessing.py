import os
import numpy as np
from PIL import Image
from skimage import exposure, transform
from skimage.filters import gaussian
from skimage import exposure, transform

data_dir = '/content/output_directory/'
output_dir = 'Preprocessed'
os.makedirs(output_dir, exist_ok=True)
image_files = os.listdir(data_dir)
target_size = (256, 256)
for image_file in image_files:
    image_path = os.path.join(data_dir, image_file)
    ultrasound_image = np.array(Image.open(image_path))
    resized_image = transform.resize(ultrasound_image, target_size, anti_aliasing=True)
    normalized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())
    enhanced_image = exposure.equalize_adapthist(normalized_image)
    output_path = os.path.join(output_dir, image_file)
    Image.fromarray((enhanced_image * 255).astype(np.uint8)).save(output_path)

print("Preprocessing complete.")

import matplotlib.pyplot as plt
from skimage import exposure, io

preprocessed_files = ['/content/output_directory/normal3_augmented_1.jpg', '/content/output_directory/normal5_augmented_7.jpg']

for i, img_path in enumerate(preprocessed_files, start=1):
    img = io.imread(img_path)
    equalized_img = exposure.equalize_hist(img)
    hist_original, bins_original = exposure.histogram(img)
    hist_equalized, bins_equalized = exposure.histogram(equalized_img)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, i*2-1)
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.6)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title(f'Image {i} Original Histogram')
    plt.subplot(2, 2, i*2)
    plt.hist(equalized_img.ravel(), bins=64, range=(0.0, 1.0), color='orange', alpha=0.7)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title(f'Image {i} Equalized Histogram')

    plt.tight_layout()

plt.show()

#convert to grayscale

from PIL import Image
import os

input_directory = '/content/Preprocessed'
output_directory = '/content/Preprocessed_Gray'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
files = os.listdir(input_directory)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(input_directory, file)
        image = Image.open(image_path)
        gray_image = image.convert('L')
        output_path = os.path.join(output_directory, file)
        gray_image.save(output_path)
print("Conversion to grayscale complete.")

from PIL import Image
import os
from skimage.filters import gaussian

input_directory = '/content/Preprocessed_Gray'
output_directory_smooth = '/content/Preprocessed_Gray_Smoothed'

if not os.path.exists(output_directory_smooth):
    os.makedirs(output_directory_smooth)

files = os.listdir(input_directory)

for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(input_directory, file)
        image = Image.open(image_path)
        smoothed_image = gaussian(np.array(image), sigma=1.5)
        smoothed_image = Image.fromarray((smoothed_image * 255).astype(np.uint8))
        output_path_smooth = os.path.join(output_directory_smooth, file)
        smoothed_image.save(output_path_smooth)

print("Smoothing process complete.")

from skimage import io, exposure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

preprocessed_files = ['/content/Preprocessed_Gray_Smoothed/sub normal 2_augmented_6.jpg', '/content/Preprocessed_Gray_Smoothed/para normal 3_augmented_7.jpg']

def plot_histogram_with_peaks(image_path):
    img = io.imread(image_path)
    hist, bins = exposure.histogram(img)
    peaks, _ = find_peaks(hist, distance=20)
    plt.figure(figsize=(8, 4))
    plt.plot(bins, hist, color='blue')
    plt.plot(bins[peaks], hist[peaks], "x", color='red', markersize=8)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram with Peaks')
    plt.show()

for img_path in preprocessed_files:
    plot_histogram_with_peaks(img_path)

from skimage import exposure, transform

data_dir = '/content/output_directory/'
output_dir = 'Swollen_Preprocessed'
os.makedirs(output_dir, exist_ok=True)
image_files = os.listdir(data_dir)
target_size = (256, 256)
for image_file in image_files:
    image_path = os.path.join(data_dir, image_file)
    ultrasound_image = np.array(Image.open(image_path))
    resized_image = transform.resize(ultrasound_image, target_size, anti_aliasing=True)
    normalized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())
    enhanced_image = exposure.equalize_adapthist(normalized_image)
    output_path = os.path.join(output_dir, image_file)
    Image.fromarray((enhanced_image * 255).astype(np.uint8)).save(output_path)

print("Preprocessing complete.")

import matplotlib.pyplot as plt
from skimage import exposure, io

preprocessed_files = ['/content/output_directory/sialadentis2_augmented_5.jpg', '/content/Swollen_Preprocessed/srojen5_augmented_6.jpg']

for i, img_path in enumerate(preprocessed_files, start=1):
    img = io.imread(img_path)
    equalized_img = exposure.equalize_hist(img)
    hist_original, bins_original = exposure.histogram(img)
    hist_equalized, bins_equalized = exposure.histogram(equalized_img)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, i*2-1)
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.6)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title(f'Image {i} Original Histogram')
    plt.subplot(2, 2, i*2)
    plt.hist(equalized_img.ravel(), bins=64, range=(0.0, 1.0), color='orange', alpha=0.7)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title(f'Image {i} Equalized Histogram')

    plt.tight_layout()

plt.show()

from PIL import Image
import os

input_directory = '/content/Swollen_Preprocessed'
output_directory = '/content/Swollen_Preprocessed_Gray'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
files = os.listdir(input_directory)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(input_directory, file)
        image = Image.open(image_path)
        gray_image = image.convert('L')
        output_path = os.path.join(output_directory, file)
        gray_image.save(output_path)

print("Conversion to grayscale complete.")

preprocessed_files = ['/content/Swollen_Preprocessed_Gray/Parotitis26_3_augmented_1.jpg', '/content/Swollen_Preprocessed_Gray/chpara10_1_augmented_6.jpg']
for i, img_path in enumerate(preprocessed_files, start=1):
    img = io.imread(img_path)
    flattened_img = img.flatten()
    plt.figure(figsize=(6, 6))
    plt.scatter(range(len(flattened_img)), flattened_img, s=0.5, color='blue')
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Intensity')
    plt.title(f'Scatter Plot for Image {i}')
    plt.show()
  from PIL import Image
import os
from skimage.filters import gaussian

input_directory = '/content/Swollen_Preprocessed_Gray'
output_directory_smooth = '/content/Swollen_Preprocessed_Gray_Smoothed'

if not os.path.exists(output_directory_smooth):
    os.makedirs(output_directory_smooth)
files = os.listdir(input_directory)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        image_path = os.path.join(input_directory, file)
        image = Image.open(image_path)
        smoothed_image = gaussian(np.array(image), sigma=1.5)
        smoothed_image = Image.fromarray((smoothed_image * 255).astype(np.uint8))

        output_path_smooth = os.path.join(output_directory_smooth, file)
        smoothed_image.save(output_path_smooth)

print("Smoothing process complete.")

from skimage import io, exposure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
preprocessed_files = ['/content/Swollen_Preprocessed_Gray_Smoothed/inhomo sj10_augmented_3.jpg', '/content/Swollen_Preprocessed_Gray_Smoothed/mand 3_augmented_3.jpg']
def plot_histogram_with_peaks(image_path):
    img = io.imread(image_path)
    hist, bins = exposure.histogram(img)

    peaks, _ = find_peaks(hist, distance=20)

    plt.figure(figsize=(8, 4))
    plt.plot(bins, hist, color='blue')
    plt.plot(bins[peaks], hist[peaks], "x", color='red', markersize=8)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram with Peaks')
    plt.show()

for img_path in preprocessed_files:
    plot_histogram_with_peaks(img_path)
