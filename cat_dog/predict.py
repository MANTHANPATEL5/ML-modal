import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Test image
image_path = "test.jpg"

# Open image
img = Image.open(image_path).convert("RGB")

# Resize
img = img.resize((128, 128))

# Convert to NumPy array
img_array = np.array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array, verbose=0)[0][0]

# Result
if prediction >= 0.5:
    print("Prediction: DOG")
    print(f"Confidence: {prediction * 100:.2f}%")
else:
    print("Prediction: CAT")
    print(f"Confidence: {(1 - prediction) * 100:.2f}%")
