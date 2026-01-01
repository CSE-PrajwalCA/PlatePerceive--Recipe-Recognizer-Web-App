# Recipe Recognizer Project

## Overview
The Recipe Recognizer project is a web application that allows users to upload a recipe image, which is then analyzed by a trained machine learning model to predict the recipe. The application provides the predicted recipe's details, including ingredients, recipe steps, calories, and nutritional information.

## Features
- **User Authentication**: Users can sign up, log in, and access personalized data.
- **Recipe Recognition**: Users can upload an image of a recipe, and the model will predict the recipe name.
- **Recipe Details**: The system provides the recipe's ingredients, steps, calories, and nutrients.
- **MongoDB Database**: Stores user credentials and recipe details.
  

## Technologies Used
- **Backend**: Flask
- **Machine Learning**: Keras, TensorFlow
- **Frontend**: HTML, CSS, JavaScript (with Flask for templating)
- **Database**: MongoDB
- **File Storage**: Local file system for image uploads

## Setup Instructions
1. **Clone the Repository**:
    ```
    git clone <repository_url>
    cd Recipe_miniproject
    ```

2. **Install Dependencies**:
    - It is recommended to create a virtual environment before installing dependencies:
      ```
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```

    - Install the required dependencies:
      ```
      pip install -r requirements.txt
      ```

3. **MongoDB Setup**:
    - Make sure MongoDB is installed and running. Create a database called `recipe_database` and a collection called `recipes`.

4. **Running the Application**:
    - To start the Flask server, run the following:
      ```
      python src/backend/app.py
      ```

    - Open the browser and navigate to `http://127.0.0.1:5000` to use the app.

## Model Training
The project includes pre-trained models in the `saved_models` directory. However, if you want to retrain the model, you can do so by following the steps in the `model_training.ipynb` notebook.

## Contributions
Feel free to fork and contribute to the project by submitting a pull request.

## License
This project is licensed under the MIT License.
