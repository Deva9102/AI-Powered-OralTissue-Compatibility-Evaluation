import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def generate_grad_cam(model, img_path, last_conv_layer_index=-3, output_path='grad_cam.jpg', alpha=0.6):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(index=last_conv_layer_index).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = tf.squeeze(heatmap).numpy()

    original_img = np.array(img).astype("uint8")
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), alpha, heatmap_colored, 1 - alpha, 0)
    cv2.imwrite(output_path, superimposed_img)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()
 generate_grad_cam(final_model_with_grad_cam, 'path/to/image.jpg')

def classify_image(img_path, model, label_df_path='directory_annotations.csv'):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    df = pd.read_csv(label_df_path)
    label_map = df['Label'].unique() 
    predicted_label = label_map[predicted_class]

    if predicted_label == 0:
        return "Prediction: Suitable For Implant"
    elif predicted_label == 1:
        return "Prediction: Not Suitable For Implant"
    else:
        return "Invalid prediction"
