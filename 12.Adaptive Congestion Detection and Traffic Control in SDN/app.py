import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from ml_models import MLModelManager
from utils import allowed_file
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize ML Model Manager
ml_manager = MLModelManager()

@app.route('/')
def home():
    """Home page with project overview."""
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for dataset."""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process dataset
            df = pd.read_csv(filepath)
            session['dataset_path'] = filepath
            session['dataset_shape'] = df.shape
            flash(f'Dataset uploaded successfully! Shape: {df.shape}', 'success')
            
            # Process the data
            ml_manager.load_data(filepath)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('home'))
    else:
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('home'))
    
    return redirect(url_for('eda'))

@app.route('/eda')
def eda():
    """EDA page with exploratory data analysis."""
    if 'dataset_path' not in session:
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('home'))
    
    try:
        # Generate EDA plots
        eda_plots = ml_manager.generate_eda_plots()
        return render_template('eda.html', plots=eda_plots, dataset_shape=session.get('dataset_shape'))
    except Exception as e:
        flash(f'Error generating EDA plots: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train selected models."""
    if 'dataset_path' not in session:
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('home'))
    
    selected_models = request.form.getlist('models')
    if not selected_models:
        flash('Please select at least one model to train.', 'error')
        return redirect(url_for('eda'))
    
    try:
        # Train models
        training_results = ml_manager.train_models(selected_models)
        session['trained_models'] = selected_models
        flash(f'Models trained successfully: {", ".join(selected_models)}', 'success')
    except Exception as e:
        flash(f'Error training models: {str(e)}', 'error')
        return redirect(url_for('eda'))
    
    return redirect(url_for('classification_performance'))

@app.route('/classification')
def classification_performance():
    """Classification performance page."""
    if 'trained_models' not in session:
        flash('Please train models first.', 'error')
        return redirect(url_for('eda'))
    
    try:
        # Get classification results
        classification_results = ml_manager.get_classification_results()
        return render_template('classification.html', results=classification_results)
    except Exception as e:
        flash(f'Error loading classification results: {str(e)}', 'error')
        return redirect(url_for('eda'))

@app.route('/regression')
def regression_performance():
    """Regression performance page."""
    if 'trained_models' not in session:
        flash('Please train models first.', 'error')
        return redirect(url_for('eda'))
    
    try:
        # Get regression results
        regression_results = ml_manager.get_regression_results()
        return render_template('regression.html', results=regression_results)
    except Exception as e:
        flash(f'Error loading regression results: {str(e)}', 'error')
        return redirect(url_for('eda'))

@app.route('/prediction')
def prediction():
    """Prediction page with input form."""
    if 'trained_models' not in session:
        flash('Please train models first.', 'error')
        return redirect(url_for('eda'))
    
    # Get feature names for form
    feature_names = ml_manager.get_feature_names()
    trained_models = session.get('trained_models', [])
    
    return render_template('prediction.html', feature_names=feature_names, trained_models=trained_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    if 'trained_models' not in session:
        flash('Please train models first.', 'error')
        return redirect(url_for('eda'))
    
    try:
        # Get form data
        input_data = {}
        feature_names = ml_manager.get_feature_names()
        
        for feature in feature_names:
            value = request.form.get(feature)
            if value:
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    input_data[feature] = value
        
        selected_model = request.form.get('model_selection', 'SVM')
        
        # Make predictions
        predictions = ml_manager.predict(input_data, selected_model)
        
        return render_template('prediction.html', 
                             feature_names=feature_names,
                             trained_models=session.get('trained_models', []),
                             predictions=predictions,
                             input_data=input_data,
                             selected_model=selected_model)
    
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('prediction'))

@app.route('/upload_test_data', methods=['POST'])
def upload_test_data():
    """Handle test data upload for batch prediction."""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('prediction'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('prediction'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load test data and make predictions
            test_df = pd.read_csv(filepath)
            selected_model = request.form.get('model_selection', 'SVM')
            
            batch_predictions = ml_manager.predict_batch(test_df, selected_model)
            
            return render_template('prediction.html',
                                 feature_names=ml_manager.get_feature_names(),
                                 trained_models=session.get('trained_models', []),
                                 batch_predictions=batch_predictions,
                                 selected_model=selected_model)
        
        except Exception as e:
            flash(f'Error processing test file: {str(e)}', 'error')
            return redirect(url_for('prediction'))
    else:
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('prediction'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
