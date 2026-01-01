import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = r"C:\Users\acess\OneDrive\Documents\Python Projects\Recipe_miniproject\src\backend\saved_models\vgg19_recipe_recognizer_optimized.h5"

model = tf.keras.models.load_model(model_path)
# classes = ['burger', 'chocolate-cake', 'dosa', 'french-fries', 'hot-dog', 'idli', 'kabab', 'pizza','Pulao', 'Samosa', 'sandwitches', 'strawberry-cake', 'tomato-soup', 'Vada', ]
classes=['Dosa', 'Idli', 'Pulao', 'Samosa', 'Vada', 'burger', 'chocolate-cake', 'french-fries', 'hot-dog', 'kabab', 'pizza', 'sandwitches', 'strawberry-cake', 'tomato-soup']

def predict_recipe(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust to your model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions from the model
    predictions = model.predict(img_array)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions[0])

    # Retrieve the confidence score for the predicted class
    confidence = predictions[0][predicted_class_index]

    # Return the predicted class label and its confidence
    return classes[predicted_class_index], float(confidence)


