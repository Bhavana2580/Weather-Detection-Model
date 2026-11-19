import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)

# -----------------------------
# LOAD DATASET
# -----------------------------
train_data = datagen.flow_from_directory(
    "Dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "Dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("\nClass Mapping:")
print(train_data.class_indices)

# -----------------------------
# LOAD MOBILENETV2
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tuning
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# -----------------------------
# CUSTOM CLASSIFICATION HEAD
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

predictions = Dense(
    train_data.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# EARLY STOPPING
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop]
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("models/weather_model.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# FINAL RESULTS
# -----------------------------
best_acc = max(history.history["val_accuracy"])

print(f"Best Validation Accuracy: {best_acc*100:.2f}%")