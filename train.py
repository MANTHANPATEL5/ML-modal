import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# SETTINGS
# ==========================================================

dataset_path = "dataset"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20

# ==========================================================
# LOAD DATASET
# ==========================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_data.class_names)

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# ==========================================================
# MOBILE NET V2
# ==========================================================

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================================
# MODEL
# ==========================================================

model = models.Sequential([

    layers.Input(shape=(160, 160, 3)),

    # Data augmentation
    data_augmentation,

    # MobileNetV2 preprocessing
    layers.Rescaling(1.0 / 127.5, offset=-1),

    # Pretrained MobileNetV2
    base_model,

    # Classification layers
    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================================
# MODEL SUMMARY
# ==========================================================

model.summary()

# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================================================
# TRAIN
# ==========================================================

print("\nStarting training...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save("cat_dog_model.keras")

print("\n====================================")
print("Model saved successfully!")
print("File: cat_dog_model.keras")
print("====================================")