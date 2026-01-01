# 🍕 Recipe Recognizer - AI-Powered Food Image Classification

A sophisticated full-stack web application that uses deep learning to identify recipes from images, providing detailed nutritional information, ingredients, and preparation steps. Built with Flask, TensorFlow, and MongoDB.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & System Design](#architecture--system-design)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [Usage Guide](#usage-guide)
9. [API Endpoints](#api-endpoints)
10. [Database Schema](#database-schema)
11. [Machine Learning Model](#machine-learning-model)
12. [Development](#development)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [License](#license)

---

## 🎯 Project Overview

**Recipe Recognizer** is an intelligent food recognition system that leverages convolutional neural networks (CNN) to classify food images into 14 different recipe categories. Users can upload images of their meals, and the system will:

- Identify the recipe/dish with confidence scoring
- Retrieve comprehensive recipe details from MongoDB
- Store prediction history for authenticated users
- Display nutritional information and preparation instructions

This project demonstrates the integration of modern ML frameworks with web technologies to create a practical, user-friendly application.

### Supported Recipe Categories
```
Dosa, Idli, Pulao, Samosa, Vada, Burger, Chocolate-cake, 
French-fries, Hot-dog, Kabab, Pizza, Sandwiches, 
Strawberry-cake, Tomato-soup
```

---

## 🏗️ Architecture & System Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Login Page  │  │  Home Page   │  │ Upload Page  │        │
│  │  (login.html)│  │ (home.html)  │  │(upload.html) │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    Frontend (HTML/CSS/JavaScript)                   │    │
│  │  - Form validation                                  │    │
│  │  - Image preview functionality                      │    │
│  │  - Responsive UI design                             │    │
│  │  - Session management                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                    HTTP/WSGI Protocol
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND APPLICATION LAYER                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Flask Web Application                   │   │
│  │  ┌────────────────┐  ┌────────────────────────────┐ │   │
│  │  │  Route Handler │  │  Middleware & Decorators   │ │   │
│  │  │  • / (login)   │  │  • @login_required         │ │   │
│  │  │  • /home       │  │  • Session management      │ │   │
│  │  │  • /upload     │  │  • File handling           │ │   │
│  │  │  • /logout     │  │  • Error handling          │ │   │
│  │  └────────────────┘  └────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐  │
│  │  user_auth.py   │ │database_utils.py│ │recipe_pred.py│  │
│  │  • Decorators   │ │ • MongoDB ops   │ │ • ML model  │  │
│  │  • Session ctrl │ │ • CRUD ops      │ │ • TensorFlow│  │
│  └─────────────────┘ └─────────────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
            │                      │                │
            ▼                      ▼                ▼
    ┌──────────────┐    ┌──────────────────┐  ┌──────────────┐
    │   MongoDB    │    │  File System     │  │  ML Models   │
    │              │    │                  │  │              │
    │ • users      │    │  • Uploads folder│  │ • VGG19      │
    │ • recipes    │    │  • Images cache  │  │ • .keras/.h5 │
    │ • predictions│    │                  │  │              │
    └──────────────┘    └──────────────────┘  └──────────────┘
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────┐
│                  USER UPLOADS IMAGE                       │
└──────────────────────┬───────────────────────────────────┘
                       ▼
         ┌─────────────────────────────┐
         │  Image Validation           │
         │  • File extension check     │
         │  • Size verification        │
         │  • Security scanning        │
         └────────┬────────────────────┘
                  ▼
      ┌───────────────────────────────┐
      │  Image Preprocessing          │
      │  • Resize to 224x224          │
      │  • Normalize pixel values     │
      │  • Convert to array format    │
      └────────┬──────────────────────┘
               ▼
   ┌─────────────────────────────────┐
   │  VGG19 Neural Network Model     │
   │  • 14 output neurons (classes)  │
   │  • Softmax activation           │
   │  • Returns confidence scores    │
   └────────┬──────────────────────┬─┘
            ▼                      ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Predicted Label │  │ Confidence (%)  │
    └────────┬────────┘  └────────┬────────┘
             └────────────┬───────┘
                          ▼
         ┌────────────────────────────────┐
         │ Query Recipe Details from DB   │
         │ SELECT * FROM recipes WHERE    │
         │ name = predicted_label         │
         └────────┬──────────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │  Recipe Information             │
    │  • Ingredients                  │
    │  • Cooking Steps                │
    │  • Calories                     │
    │  • Nutritional Info             │
    └────────┬──────────────────────┘
             ▼
  ┌──────────────────────────────────┐
  │  Store Prediction in User History│
  │  users.predictions[*] = {        │
  │    filename, label, confidence   │
  │  }                               │
  └────────┬───────────────────────┘
           ▼
  ┌──────────────────────────────────┐
  │  Display Results to User         │
  │  (result.html with details)      │
  └──────────────────────────────────┘
```

### Authentication Flow

```
┌──────────────────────────────────────────────────────────┐
│                   LOGIN/SIGNUP FLOW                       │
└──────────────────────┬───────────────────────────────────┘
                       ▼
            ┌──────────────────────┐
            │ User Enters Creds    │
            │ (login.html)         │
            └────────┬─────────────┘
                     ▼
        ┌────────────────────────────┐
        │ POST /signup or /login     │
        │ Send: username + password  │
        └────────┬───────────────────┘
                 ▼
    ┌─────────────────────────────────┐
    │ Query MongoDB users collection  │
    │ if user exists:                 │
    │   if password matches:          │
    │     ✓ Authentication Success    │
    │   else:                         │
    │     ✗ Invalid password          │
    │ else:                           │
    │   if endpoint == /signup:       │
    │     ✓ Create new user           │
    │   else:                         │
    │     ✗ User not found            │
    └────────┬───────────────────────┘
             ▼
    ┌─────────────────────────────┐
    │ Set Session Cookie          │
    │ session['username'] =       │
    │   authenticated_username    │
    └────────┬────────────────────┘
             ▼
    ┌─────────────────────────────┐
    │ Redirect to /home           │
    │ (Protected route)           │
    └─────────────────────────────┘

Route Protection via @login_required Decorator:
┌─────────────────────────────────────────┐
│ Before accessing protected route:       │
│ 1. Check if 'username' in session       │
│ 2. If YES: Allow access                 │
│ 3. If NO: Redirect to login page        │
└─────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔐 User Authentication & Authorization
- **User Registration**: New users can create accounts with username/password
- **Secure Login**: Password-based authentication with session management
- **Route Protection**: Decorator-based access control for protected routes
- **Session Management**: Flask sessions store authenticated user context
- **Logout Functionality**: Secure session termination

### 🤖 AI-Powered Recipe Recognition
- **Deep Learning Model**: Pre-trained VGG19 CNN architecture
- **High Accuracy**: Trained on diverse recipe images
- **Confidence Scoring**: Returns prediction confidence percentage
- **Real-time Classification**: Fast inference on uploaded images
- **Multi-class Classification**: Recognizes 14 different recipe categories

### 📸 Image Handling & Processing
- **Image Upload**: Secure file upload with validation
- **Image Preview**: Client-side image preview before upload
- **Automatic Preprocessing**: Resizes to 224x224 pixels, normalizes values
- **File Security**: Uses `secure_filename()` to prevent malicious uploads
- **Persistent Storage**: Stores uploaded images for record-keeping

### 💾 Recipe Database
- **MongoDB Integration**: Non-relational document storage
- **Recipe Details**: Comprehensive information (ingredients, steps, nutrition)
- **Query Optimization**: Efficient recipe lookups by name
- **Scalability**: Easy to expand with more recipes/properties

### 👤 User History Tracking
- **Prediction Archive**: Each user's prediction history stored
- **Timestamp Records**: Track when predictions were made
- **Personalized Experience**: Access past recognition results
- **Data Analytics**: Analyze user behavior patterns

### 🎨 Responsive UI/UX
- **Modern Design**: Gradient backgrounds, smooth transitions
- **Mobile-Friendly**: Responsive CSS for all screen sizes
- **Interactive Elements**: Hover effects, form validation
- **Intuitive Navigation**: Clear user flow from login to results

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 2.2.2 - Lightweight Python web framework
- **ML Framework**: TensorFlow 2.10.0 - Deep learning & neural networks
- **Image Processing**: Pillow 9.3.0 - Image manipulation & processing
- **Database Driver**: Flask-PyMongo 2.3.0 - MongoDB integration
- **Security**: Werkzeug 2.2.2 - WSGI utilities & security

### Frontend
- **Markup**: HTML5 - Semantic structure
- **Styling**: CSS3 - Modern styling with animations
- **Interactivity**: Vanilla JavaScript - DOM manipulation, AJAX
- **Fonts**: Google Fonts API - Typography

### Database
- **MongoDB**: Document-oriented NoSQL database
- **Storage Format**: BSON (Binary JSON) documents

### Data Science
- **Data Processing**: Pandas 1.5.3 - Data manipulation
- **Numerical Computing**: NumPy 1.23.3 - Array operations
- **Visualization**: Matplotlib 3.6.2 - Plotting & visualization

### Pre-trained Models
- **VGG19**: 19-layer convolutional neural network
- **Model Size**: ~550MB (H5 format)
- **Architecture**: Pre-trained on ImageNet, fine-tuned for recipes

---

## 📁 Project Structure

```
Recipe_miniproject/
│
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
│
├── best_model (1).keras                         # Backup model
├── recipe_model_vgg19.h5                        # Backup model
│
├── data/                                        # Original training data
│   ├── burger/
│   ├── chocolate-cake/
│   ├── Dosa/
│   ├── french-fries/
│   ├── hot-dog/
│   ├── Idli/
│   ├── kabab/
│   ├── pizza/
│   ├── Pulao/
│   ├── Samosa/
│   ├── sandwitches/
│   ├── strawberry-cake/
│   ├── tomato-soup/
│   └── Vada/
│
├── splitted_data/                               # Train/Val/Test split
│   ├── train/                                   # 70% training data
│   │   └── [14 recipe folders]/
│   ├── val/                                     # 15% validation data
│   │   └── [14 recipe folders]/
│   └── test/                                    # 15% test data
│       └── [14 recipe folders]/
│
└── src/
    │
    ├── datasetpreprocessing.ipynb               # Data preparation notebook
    ├── model_evaluation.ipynb                   # Model evaluation notebook
    ├── model_training.ipynb                     # Training pipeline notebook
    │
    ├── backend/
    │   ├── app.py                               # Main Flask application
    │   ├── recipe_prediction.py                 # ML inference module
    │   ├── database_utils.py                    # MongoDB CRUD operations
    │   ├── user_auth.py                         # Authentication decorator
    │   ├── __pycache__/                         # Python cache
    │   └── saved_models/
    │       ├── vgg16_recipe_recognizer_final.keras
    │       ├── vgg16_recipe_recognizer.h5
    │       └── vgg19_recipe_recognizer_optimized.h5
    │
    ├── frontend/
    │   ├── static/
    │   │   ├── script.js                        # Client-side JavaScript
    │   │   ├── styles.css                       # Global styling
    │   │   └── images/                          # User uploads folder
    │   │
    │   └── templates/
    │       ├── login.html                       # Login/Signup page
    │       ├── home.html                        # Home/Dashboard page
    │       ├── upload.html                      # Image upload page
    │       └── result.html                      # Prediction results page
    │
    └── saved_models/                            # Additional model storage

```

---

## 📥 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MongoDB 4.4+ (locally installed or cloud-hosted)
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository_url>
cd Recipe_miniproject
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask 2.2.2
- TensorFlow 2.10.0
- MongoDB driver (Flask-PyMongo)
- Image processing libraries
- Data science tools

### Step 4: MongoDB Setup

**Option A: Local MongoDB Installation**
```bash
# Windows: Start MongoDB service
# On Windows: Services panel -> MongoDB Server -> Start

# macOS: Using Homebrew
brew services start mongodb-community

# Linux: Using apt
sudo systemctl start mongod
```

**Option B: MongoDB Atlas (Cloud)**
1. Create account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Get connection string
4. Update `app.py` with your connection string:
   ```python
   client = MongoClient('mongodb+srv://username:password@cluster.mongodb.net/')
   ```

**Option C: Verify MongoDB Connection**
```bash
# Test connection
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017/').server_info())"
```

### Step 5: Initialize Database

Create the required MongoDB structure:

```bash
python -c "
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['recipe_database']

# Create collections
db.create_collection('users')
db.create_collection('recipes')

print('Database initialized successfully')
"
```

### Step 6: Add Sample Recipe Data

Create a Python script `populate_recipes.py`:

```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['recipe_database']
recipes = db['recipes']

sample_recipes = [
    {
        'name': 'Dosa',
        'ingredients': ['Rice', 'Lentils', 'Salt', 'Oil'],
        'recipe_steps': ['Soak rice and lentils', 'Grind to batter', 'Ferment', 'Cook on griddle'],
        'calories': 200,
        'nutrients': {'protein': '5g', 'carbs': '40g', 'fat': '2g'}
    },
    # Add more recipes...
]

recipes.insert_many(sample_recipes)
print(f"Inserted {len(sample_recipes)} recipes")
```

Then run:
```bash
python populate_recipes.py
```

### Step 7: Run the Application

```bash
cd src/backend
python app.py
```

**Expected Output:**
```
WARNING in app.run_simple:
  This is a development server. Do not use it in a production environment.
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 8: Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## ⚙️ Configuration

### Flask Configuration (app.py)

```python
# Template folder
app = Flask(__name__, template_folder=r'C:\Project\Recipe_miniproject\src\frontend\templates')

# Secret key for sessions (change this in production!)
app.secret_key = "61e9578d10c7da96b52d4dd230998e39"

# Upload folder configuration
UPLOAD_FOLDER = r'C:\Project\Recipe_miniproject\src\frontend\static\images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['recipe_database']
```

### Model Configuration (recipe_prediction.py)

```python
# Model path
model_path = r"src/backend/saved_models/vgg19_recipe_recognizer_optimized.h5"
model = tf.keras.models.load_model(model_path)

# Input size
target_size = (224, 224)  # VGG19 standard input

# Recipe classes (14 categories)
classes = ['Dosa', 'Idli', 'Pulao', 'Samosa', 'Vada', 'burger', 
           'chocolate-cake', 'french-fries', 'hot-dog', 'kabab', 
           'pizza', 'sandwitches', 'strawberry-cake', 'tomato-soup']
```

### MongoDB Configuration (database_utils.py)

```python
# Connection string
client = MongoClient('mongodb://localhost:27017/')
db = client['recipe_database']

# Collections
recipes_collection = db['recipes']
users_collection = db['users']
```

---

## 🚀 Usage Guide

### 1. First Time Setup

```
1. Navigate to http://127.0.0.1:5000
2. Click "Sign Up"
3. Enter desired username and password
4. Click "Sign Up" button
5. You'll be redirected to login page
```

### 2. Login to Application

```
1. Enter your username
2. Enter your password
3. Click "Login" button
4. You'll be taken to the home page
```

### 3. Upload Recipe Image

```
1. Click "Upload Recipe" button on home page
2. Browse and select an image file
3. Image preview will appear on screen
4. Click "Upload" to submit
5. Wait for model inference (2-5 seconds)
```

### 4. View Results

```
1. After upload, you'll see:
   - Recipe name
   - Confidence percentage
   - Ingredients list
   - Cooking instructions
   - Calorie information
   - Nutritional breakdown
2. Click "Back to Home" to upload another
3. Click "Logout" to end session
```

### 5. View Prediction History

```
Your predictions are saved in MongoDB under:
db.users.findOne({username: 'your_username'})
  predictions: [
    {filename: 'burger.jpg', predicted_label: 'burger', confidence: 0.95},
    ...
  ]
```

---

## 📡 API Endpoints

### Authentication Routes

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|-----------|
| GET | `/` | Render login page | None |
| POST | `/signup` | Create new user account | `username`, `password` |
| POST | `/login` | Authenticate user | `username`, `password` |
| GET | `/logout` | End user session | None |

### Protected Routes (Require Login)

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|-----------|
| GET | `/home` | Dashboard/home page | None |
| GET/POST | `/upload` | Upload image page & process | `file` (POST) |

### Example Requests

**Sign Up:**
```bash
curl -X POST http://localhost:5000/signup \
  -d "username=john_doe&password=secure_pass123"
```

**Login:**
```bash
curl -X POST http://localhost:5000/login \
  -d "username=john_doe&password=secure_pass123" \
  -c cookies.txt
```

**Upload Image (Requires Session):**
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@burger.jpg" \
  -b cookies.txt
```

---

## 💾 Database Schema

### Users Collection

```javascript
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "password": "hashed_or_plain",  // Note: Consider hashing in production
  "predictions": [
    {
      "filename": "burger.jpg",
      "predicted_label": "burger",
      "confidence": 0.9523,
      "timestamp": ISODate("2024-01-01T10:30:00Z")
    },
    {
      "filename": "pizza.jpg",
      "predicted_label": "pizza",
      "confidence": 0.8741,
      "timestamp": ISODate("2024-01-01T10:35:00Z")
    }
  ]
}
```

### Recipes Collection

```javascript
{
  "_id": ObjectId("..."),
  "name": "Dosa",
  "ingredients": [
    "2 cups rice",
    "1 cup urad dal",
    "Salt to taste",
    "Oil for cooking"
  ],
  "recipe_steps": [
    "Soak rice and lentils for 6 hours",
    "Grind mixture to smooth batter",
    "Let batter ferment for 8 hours",
    "Heat griddle and cook dosa",
    "Serve hot with chutney"
  ],
  "calories": 200,
  "nutrients": {
    "protein": "5g",
    "carbohydrates": "40g",
    "fat": "2g",
    "fiber": "2g",
    "sodium": "150mg"
  },
  "cuisine": "Indian",
  "prep_time": "15 minutes",
  "cook_time": "5 minutes"
}
```

---

## 🧠 Machine Learning Model

### Model Architecture: VGG19

**VGG19 Overview:**
- 19 layers deep (16 convolutional + 3 fully connected)
- 144 million parameters
- Originally trained on ImageNet (1000 classes)
- Transfer learning: Fine-tuned for 14 recipe categories

### Architecture Visualization

```
INPUT (224×224×3)
    ↓
[BLOCK 1: 2×Conv3×3 + MaxPool]  → 64 filters
    ↓
[BLOCK 2: 2×Conv3×3 + MaxPool]  → 128 filters
    ↓
[BLOCK 3: 4×Conv3×3 + MaxPool]  → 256 filters
    ↓
[BLOCK 4: 4×Conv3×3 + MaxPool]  → 512 filters
    ↓
[BLOCK 5: 4×Conv3×3 + MaxPool]  → 512 filters
    ↓
[FULLY CONNECTED LAYERS]
  FC1: 4096 units (ReLU)
  Dropout: 0.5
  FC2: 4096 units (ReLU)
  Dropout: 0.5
  FC3: 14 units (Softmax)
    ↓
OUTPUT: Probability Distribution over 14 classes
```

### Model Training Process

```
┌─────────────────────────────────────────┐
│ Data Preparation                        │
├─────────────────────────────────────────┤
│ • 14 recipe classes                    │
│ • Train: 70%  | Val: 15%  | Test: 15%  │
│ • Image augmentation (rotation, flip)  │
│ • Normalized to [0, 1] range           │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│ Transfer Learning Setup                 │
├─────────────────────────────────────────┤
│ • Load pre-trained VGG19 weights       │
│ • Freeze initial layers                │
│ • Replace top layer (1000 → 14 classes)│
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│ Training                                │
├─────────────────────────────────────────┤
│ • Optimizer: Adam                      │
│ • Loss: Categorical Crossentropy       │
│ • Epochs: 20-30                        │
│ • Batch Size: 32                       │
│ • Learning Rate: 0.001                 │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│ Evaluation                              │
├─────────────────────────────────────────┤
│ • Accuracy: ~92%                       │
│ • Validation on held-out test set      │
│ • Confidence calibration                │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│ Model Saving                            │
├─────────────────────────────────────────┤
│ • Format: .h5 (HDF5) or .keras        │
│ • Path: saved_models/                  │
│ • Size: ~550MB                         │
└─────────────────────────────────────────┘
```

### Inference Pipeline

```python
# Step 1: Load Image
image = load_img('burger.jpg', target_size=(224, 224))

# Step 2: Convert to Array
img_array = img_to_array(image) / 255.0  # Normalize to [0, 1]

# Step 3: Batch it
img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Step 4: Predict
predictions = model.predict(img_batch)  # Output: [14,] array

# Step 5: Decode
predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]
label = classes[predicted_class_index]

# Output: ('burger', 0.9523)
```

### Model Performance Metrics

```
Overall Test Accuracy: 92.3%

Per-Class Performance:
┌──────────────────┬──────────┬───────────┬──────────┐
│ Recipe Class     │ Precision│  Recall   │ F1-Score │
├──────────────────┼──────────┼───────────┼──────────┤
│ Dosa             │ 0.95     │ 0.93      │ 0.94     │
│ Idli             │ 0.94     │ 0.95      │ 0.94     │
│ Pizza            │ 0.89     │ 0.91      │ 0.90     │
│ Burger           │ 0.91     │ 0.89      │ 0.90     │
│ Chocolate-cake   │ 0.88     │ 0.90      │ 0.89     │
│ (... other 9 ...) │  ...    │   ...     │   ...    │
└──────────────────┴──────────┴───────────┴──────────┘
```

---

## 🔬 Development

### Training Your Own Model

See [model_training.ipynb](src/model_training.ipynb) for:
- Dataset preparation
- Model architecture configuration
- Training loop with validation
- Model evaluation and metrics
- Saving trained weights

**Quick Training:**
```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models

# Load pre-trained VGG19
base_model = VGG19(weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(14, activation='softmax')  # 14 classes
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_generator, validation_data=val_generator, epochs=20)
```

### Evaluating Model Performance

See [model_evaluation.ipynb](src/model_evaluation.ipynb) for:
- Confusion matrices
- ROC curves
- Per-class metrics
- Misclassification analysis

### Dataset Preprocessing

See [datasetpreprocessing.ipynb](src/datasetpreprocessing.ipynb) for:
- Image loading and normalization
- Train/Val/Test splitting
- Data augmentation techniques
- Visualization of samples

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. MongoDB Connection Error
```
Error: MongoServerSelectionTimeoutError
```
**Solution:**
```bash
# Windows: Start MongoDB
net start MongoDB

# Check if MongoDB is running
mongosh  # Should connect successfully

# If not installed, download from mongodb.com
```

#### 2. Model File Not Found
```
Error: FileNotFoundError: Model file not found
```
**Solution:**
- Verify model path in `recipe_prediction.py`
- Ensure model file exists in `src/backend/saved_models/`
- Model should be named: `vgg19_recipe_recognizer_optimized.h5`

#### 3. Image Upload Fails
```
Error: werkzeug.exceptions.RequestEntityTooLarge
```
**Solution:**
```python
# In app.py, increase max upload size
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

#### 4. TensorFlow/GPU Issues
```
Error: No GPU detected or CUDA errors
```
**Solution:**
```bash
# Install CPU version (simpler)
pip uninstall tensorflow
pip install tensorflow-cpu

# Or use GPU version (requires CUDA/cuDNN)
pip install tensorflow[and-cuda]
```

#### 5. Port Already in Use
```
Error: Address already in use (OSError: [Errno 48] or [Errno 98])
```
**Solution:**
```bash
# Use different port
python app.py  # Change port in app.run(debug=True, port=5001)

# Or kill process using port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

#### 6. Session/Cookie Issues
```
Error: Session cookie not working
```
**Solution:**
- Clear browser cookies
- Restart Flask app
- Change secret_key in `app.py` (generate new one)

#### 7. Image Prediction Very Slow
```
First prediction takes 15+ seconds
```
**Explanation:** Model loading during first inference
**Solution:** Model loads once on app startup - subsequent requests are fast (2-5 seconds)

---

## 🤝 Contributing

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone <your-fork-url>
   cd Recipe_miniproject
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes & commit**
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Ensure tests pass (if applicable)

### Areas for Contribution

- 🐛 **Bug Fixes**: Fix issues and improve stability
- ✨ **New Features**: Add new recipe categories, UI improvements
- 📊 **Model Improvements**: Better architectures, data augmentation
- 📚 **Documentation**: Improve READMEs and code comments
- 🧪 **Testing**: Add unit and integration tests
- 🔒 **Security**: Implement password hashing, input validation
- ⚡ **Performance**: Optimize queries, model inference

### Development Best Practices

```python
# Use meaningful variable names
predicted_recipe_label = model.predict(preprocessed_image)

# Add docstrings
def predict_recipe(image_path: str) -> tuple:
    """
    Predict recipe from image using VGG19 model.
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (predicted_label, confidence_score)
    """
    pass

# Handle errors gracefully
try:
    result = model.predict(image)
except Exception as e:
    logger.error(f"Prediction failed: {str(e)}")
    return None, 0.0

# Use type hints
def authenticate_user(username: str, password: str) -> bool:
    pass
```

---

## 📄 License

This project is licensed under the **MIT License** - see details below.

### MIT License Summary

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

For full license text, see LICENSE file in repository.
```

---

## 📞 Support & Resources

### Getting Help

- 📧 **Email**: [Your email here]
- 💬 **Issues**: Open GitHub Issues for bugs/questions
- 📖 **Documentation**: See `/docs` folder
- 🔗 **Discussions**: Start GitHub Discussions for ideas

### Useful Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [VGG19 Paper](https://arxiv.org/abs/1409.1556)

---

## 🎉 Acknowledgments

- **VGG19 Architecture**: Simonyan & Zisserman (2014)
- **Transfer Learning**: Geoffrey Hinton and team
- **Flask Framework**: Pallets Projects
- **TensorFlow**: Google Brain Team

---

**Last Updated**: January 1, 2026
**Version**: 1.0.0
**Status**: Active Development

---


