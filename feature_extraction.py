def load_images(directory):
    images = []
    limit = 104
    image_files = os.listdir(directory)[:limit]

    for file_name in image_files:
        img = imread(os.path.join(directory, file_name))
        images.append(img)

    return images

swollen_images = load_images(swollen_dir)
normal_images = load_images(normal_dir)

def preprocess_images(images):
    processed_images = []
    flatten_images = []

    for img in images:
        if len(img.shape) > 2:
            img = img.mean(axis=2)
        flatten_img = img.flatten()
        processed_images.append(img)
        flatten_images.append(flatten_img)

    return np.array(processed_images), np.array(flatten_images)

swollen_processed, swollen_flatten = preprocess_images(swollen_images)
normal_processed, normal_flatten = preprocess_images(normal_images)

flatten_size_n = normal_flatten.shape[1]

swollen_df = pd.DataFrame(swollen_flatten)
normal_df = pd.DataFrame(normal_flatten)

print("Swollen DataFrame shape:", swollen_df.shape)
print("Normal DataFrame shape:", normal_df.shape)
print("Sample of Swollen DataFrame:")
print(swollen_df.head())
print("Sample of Normal DataFrames")

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    features = []
    for img in images:
        fd, _ = hog(img,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=True)
        features.append(fd)
    return np.array(features)


normal_dir = '/content/Preprocessed_Gray'
swollen_dir = '/content/Swollen_Preprocessed_Gray'
normal_flat = load_and_flatten_images(normal_dir, limit=104)
swollen_flat = load_and_flatten_images(swollen_dir, limit=104)
normal_df = pd.DataFrame(normal_flat)
swollen_df = pd.DataFrame(swollen_flat)
normal_df['label'] = 0
swollen_df['label'] = 1

normal_images = load_images(normal_dir, limit=104)
swollen_images = load_images(swollen_dir, limit=104)
normal_hog = extract_hog_features(normal_images)
swollen_hog = extract_hog_features(swollen_images)
hog_normal_df = pd.DataFrame(normal_hog)
hog_swollen_df = pd.DataFrame(swollen_hog)
hog_normal_df['label'] = 0
hog_swollen_df['label'] = 1

flat_df = pd.concat([normal_df, swollen_df], ignore_index=True)
hog_df = pd.concat([hog_normal_df, hog_swollen_df], ignore_index=True)

flat_df = shuffle(flat_df).reset_index(drop=True)
hog_df = shuffle(hog_df).reset_index(drop=True)

X_flat = flat_df.drop('label', axis=1).to_numpy()
y_flat = flat_df['label'].to_numpy()

X_hog = hog_df.drop('label', axis=1).to_numpy()
y_hog = hog_df['label'].to_numpy()

print(f"Flattened Data  -> X: {X_flat.shape}, y: {y_flat.shape}")
print(f"HOG Features    -> X: {X_hog.shape}, y: {y_hog.shape}")

import numpy as np
mean_swollen_hog = np.mean(swollen_hog_features, axis=0)
mean_normal_hog = np.mean(normal_hog_features, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(mean_swollen_hog, label='Swollen Mean HOG Features')
plt.plot(mean_normal_hog, label='Normal Mean HOG Features')
plt.xlabel('Feature Index')
plt.ylabel('Mean Value')
plt.title('Mean HOG Features Comparison')
plt.legend()
plt.show()
