nor=os.listdir('/content/drive/MyDrive/fyp/project/data/normal')
swol=os.listdir('/content/drive/MyDrive/fyp/project/data/swollen')
limit=13
normal=[None]*limit
j=0
for i in nor:
  if(j<limit):
    normal[j]=imread("/content/drive/MyDrive/fyp/project/data/normal/" +i)
    j+=1
  else:
    break
limit=104
swollen=[None]*limit
j=0
for i in swol:
  if(j<limit):
    swollen[j]=imread("/content/drive/MyDrive/fyp/project/data/swollen/" +i)
    j+=1
  else:
    break
Image_size=150
import tensorflow as tf
from tensorflow.keras import layers
resize_and_rescale=tf.keras.Sequential([layers.experimental.preprocessing.Resizing(Image_size,Image_size),layers.experimental.preprocessing.Rescaling(1./255)])
from PIL import Image
from PIL import Image, ImageFilter, ImageEnhance
input_directory = '/content/drive/MyDrive/fyp/project/data/normal'
output_directory = '/content/output_directory'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
def flip_horizontal(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_90(image):
    return image.rotate(90)

def rotate_180(image):
    return image.rotate(180)

def mirror(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

def brightness_increase(image, factor=1.2):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def contrast_increase(image, factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def sharpness_increase(image, factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

transformations = [flip_horizontal, flip_vertical, rotate_90, rotate_180,mirror,brightness_increase,contrast_increase,sharpness_increase]
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_directory, filename)
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        augmented_images = []
        for idx, transform in enumerate(transformations):
            augmented_img = transform(image)
            output_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}_augmented_{idx}.jpg')
            augmented_img.save(output_path)

print("Augmented images saved in output_directory.")

input_directory = '/content/drive/MyDrive/fyp/project/data/swollen'
output_directory = '/content/output_directory'
def flip_horizontal(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_90(image):
    return image.rotate(90)

def rotate_180(image):
    return image.rotate(180)

def mirror(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

def brightness_increase(image, factor=1.2):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def contrast_increase(image, factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def sharpness_increase(image, factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

transformations = [flip_horizontal, flip_vertical, rotate_90, rotate_180,mirror,brightness_increase,contrast_increase,sharpness_increase]

for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_directory, filename)
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        augmented_images = []
        for idx, transform in enumerate(transformations):
            augmented_img = transform(image)
            output_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}_augmented_{idx}.jpg')
            augmented_img.save(output_path)

print("Augmented images saved in output_directory.")
