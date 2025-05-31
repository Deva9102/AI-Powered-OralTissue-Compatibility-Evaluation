import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu'):
    shortcut = x

    y = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)

    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)

    if strides != (1, 1) or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([shortcut, y])
    out = layers.Activation(activation)(out)
    return out

# ---------- ResNet Model ----------
def build_resnet(input_shape=(150, 150, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 128, strides=(2, 2))
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 256, strides=(2, 2))
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="ResNetModel")

# ---------- DBN (MLP-style) Model ----------
def build_dbn(input_shape=(150, 150, 3), num_classes=2):
    model = tf.keras.Sequential(name="DBNModel")
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# ---------- Hybrid Model ----------
def build_hybrid_model(input_shape=(150, 150, 3), num_classes=2):
    input_layer = layers.Input(shape=input_shape)
    
    resnet = build_resnet(input_shape, num_classes)
    dbn = build_dbn(input_shape, num_classes)

    resnet_out = resnet(input_layer)
    dbn_out = dbn(input_layer)

    merged = layers.concatenate([resnet_out, dbn_out])
    final_output = layers.Dense(num_classes, activation='softmax')(merged)

    return Model(inputs=input_layer, outputs=final_output, name="HybridModel")

# ---------- Grad-CAM Base Model ----------
def build_grad_cam_model(resnet_model):
    grad_cam_layer = resnet_model.layers[-4].output  # Last conv layer
    return Model(inputs=resnet_model.input, outputs=grad_cam_layer, name="GradCamModel")
