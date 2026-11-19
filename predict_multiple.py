import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(
    "models/weather_model.keras"
)

class_names = [
    "Cloudy",
    "Foggy",
    "Rainy",
    "Sunny",
    "dew",
    "snow"
]

test_folder = "test_images"

for filename in os.listdir(test_folder):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        img_path = os.path.join(
            test_folder,
            filename
        )

        img = image.load_img(
            img_path,
            target_size=(224,224)
        )

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array,
            axis=0
        )
        img_array = img_array / 255.0

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = class_names[
            np.argmax(prediction)
        ]

        confidence = (
            np.max(prediction) * 100
        )

        print(
            f"{filename} --> "
            f"{predicted_class} "
            f"({confidence:.2f}%)"
        )