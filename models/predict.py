import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


MODEL_PATH = "plant_disease_detection.h5"
IMAGE_PATH = "plant.jpg"
CATEGORIES_PATH = "categories.json"


print("Building model...")

# Same input configuration as the original notebook
image_size = 224
input_shape = (224, 224, 3)


# Build the exact original MobileNetV2 architecture
base_model = keras.applications.MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=input_shape
)

base_model.trainable = False

inputs = keras.Input(shape=input_shape)

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(38, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x)


print("Loading trained weights...")

# Load only the trained weights
model.load_weights(MODEL_PATH)

print("Model loaded successfully!")


# Load class names
with open(CATEGORIES_PATH, "r") as f:
    categories = json.load(f)

class_names = {v: k for k, v in categories.items()}


# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))


# Same preprocessing used during training
image_array = np.array(image, dtype=np.float32)
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)


print("Running prediction...")

predictions = model.predict(image_array, verbose=0)

predicted_index = int(np.argmax(predictions[0]))
confidence = float(predictions[0][predicted_index])

predicted_disease = class_names[predicted_index]


print()
print("================================")
print("       PREDICTION RESULT")
print("================================")
print("Disease:", predicted_disease)
print("Confidence:", f"{confidence * 100:.2f}%")
print("================================")