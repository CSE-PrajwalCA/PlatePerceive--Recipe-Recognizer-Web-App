from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
from recipe_prediction import predict_recipe
from database_utils import fetch_recipe_details, insert_user, authenticate_user
from user_auth import login_required
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__, template_folder=r'C:\Project\Recipe_miniproject\src\frontend\templates')
app.secret_key = "61e9578d10c7da96b52d4dd230998e39"

UPLOAD_FOLDER = r'C:\Project\Recipe_miniproject\src\frontend\static\images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['recipe_database']
user_collection = db['users']

# Route for the login page
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    if insert_user(username, password):
        flash('Signup successful. Please login.')
    else:
        flash('User already exists.')
    return redirect(url_for('login'))

@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form['username']
    password = request.form['password']
    if authenticate_user(username, password):
        session['username'] = username
        return render_template('home.html')
        #return redirect(url_for('home'))  # Redirect to home route
    flash('Invalid login credentials.')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'], endpoint='file_upload')  # Use a unique endpoint
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the prediction and confidence
        predicted_label, confidence = predict_recipe(file_path)

        # Fetch recipe details
        recipe_details = fetch_recipe_details(predicted_label)

        # Save prediction in MongoDB under the logged-in user's data
        user_collection.update_one(
            {'username': session['username']},
            {'$push': {
                'predictions': {
                    'filename': filename,
                    'predicted_label': predicted_label,
                    'confidence': confidence
                }
            }},
            upsert=True
        )

        # Display the prediction results
        # return render_template('result.html', recipe=recipe_details, label=predicted_label, confidence=confidence)
        confidence_percentage = round(confidence * 100, 2)  # Convert to percentage and round to 2 decimal places
        return render_template('result.html', recipe=recipe_details, label=predicted_label, accuracy=confidence_percentage)

    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
