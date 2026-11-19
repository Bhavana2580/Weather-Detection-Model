import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

model = tf.keras.models.load_model(
    "models/weather_model.keras"
)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    "Dataset",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

loss, accuracy = model.evaluate(val_data)

print("\nRESULTS")
print("-" * 30)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

print("\nClass Mapping:")
print(val_data.class_indices)