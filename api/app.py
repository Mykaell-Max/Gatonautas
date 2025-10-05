#!/usr/bin/env python3
"""
Exoplanet Detection API

A simple API for detecting exoplanets from light curve data.
Supports both CSV file uploads and target name lookups from MAST.
"""

from flask import Flask, request, jsonify
import os
import sys
import tempfile
import traceback
from werkzeug.utils import secure_filename
from utils.hyperparameters import hyperparameter_manager

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml', 'src'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'temp')
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/look-for-exoplanet', methods=['POST'])
def look_for_exoplanet():
    """
    Main endpoint for exoplanet detection.
    
    Parameters:
    - label (optional): Target name or coordinates for MAST lookup
    - lightcurve (optional): CSV file upload with light curve data
    
    Returns:
    - JSON with extracted features and exoplanet prediction
    """
    try:
        # Get parameters
        label = request.form.get('label', '').strip()
        lightcurve_file = request.files.get('lightcurve')
        
        # Validate input
        if not label and not lightcurve_file:
            return jsonify({
                'error': 'At least one parameter is required: label or lightcurve file'
            }), 400
        
        # Determine data source and processing method
        data_source = None
        features = None
        
        if lightcurve_file and lightcurve_file.filename:
            # Process uploaded CSV file
            if not allowed_file(lightcurve_file.filename):
                return jsonify({
                    'error': 'Invalid file type. Only CSV files are allowed.'
                }), 400
            
            # Save uploaded file temporarily
            filename = secure_filename(lightcurve_file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            lightcurve_file.save(temp_path)
            
            try:
                # Import and use pipeline function
                from pipeline import extract_all_features_from_csv
                features = extract_all_features_from_csv(temp_path, verbose=True)
                data_source = 'csv_upload'
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        elif label:
            # Process target name lookup
            try:
                from pipeline import extract_all_features_v2
                features = extract_all_features_v2(label, verbose=True)
                data_source = 'mast_download'
            except Exception as e:
                return jsonify({
                    'error': f'Failed to download data for target "{label}": {str(e)}'
                }), 400
        
        if not features:
            return jsonify({
                'error': 'Failed to extract features from the provided data'
            }), 500
        
        # Get exoplanet prediction
        prediction = None
        try:
            from evaluate import predict_single_sample_api
            
            # Use the stacking model by default
            model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'artifacts', 'meta_learner.joblib')
            
            if os.path.exists(model_path):
                # Get current hyperparameters
                current_hyperparameters = hyperparameter_manager.get_hyperparameters()
                
                # Make prediction with current hyperparameters
                prediction = predict_single_sample_api(model_path, features, hyperparameters=current_hyperparameters)
            else:
                return jsonify({
                    'error': 'ML model not found. Please ensure meta_learner.joblib exists.'
                }), 500
                
        except Exception as e:
            # Return features without prediction if ML fails
            app.logger.warning(f"ML prediction failed: {str(e)}")
            prediction = {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'prediction_label': 'Unknown',
                'exoplanet_confidence': None,
                'non_exoplanet_confidence': None
            }
        
        # Prepare response
        response = {
            'features': features,
            'prediction': prediction,
            'metadata': {
                'data_source': data_source,
                'target': label if label else 'uploaded_file',
                'features_extracted': len(features)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'exoplanet-detection-api'
    })

@app.route('/hyperparameters', methods=['GET'])
def get_all_hyperparameters():
    """Get all hyperparameters for all models."""
    try:
        hyperparameters = hyperparameter_manager.get_hyperparameters()
        available_models = hyperparameter_manager.get_available_models()
        
        return jsonify({
            'hyperparameters': hyperparameters,
            'available_models': available_models,
            'note': 'Use GET /hyperparameters/<model_name> for specific model or POST /hyperparameters/<model_name> to update'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get hyperparameters: {str(e)}'}), 500

@app.route('/hyperparameters/<model_name>', methods=['GET'])
def get_hyperparameters(model_name):
    """Get hyperparameters for a specific model."""
    try:
        if model_name not in hyperparameter_manager.get_available_models():
            return jsonify({
                'error': f'Unknown model: {model_name}',
                'available_models': hyperparameter_manager.get_available_models()
            }), 400
        
        model_info = hyperparameter_manager.get_model_info(model_name)
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': f'Failed to get hyperparameters for {model_name}: {str(e)}'}), 500

@app.route('/hyperparameters/<model_name>', methods=['POST'])
def update_hyperparameters(model_name):
    """Update hyperparameters for a specific model."""
    try:
        if model_name not in hyperparameter_manager.get_available_models():
            return jsonify({
                'error': f'Unknown model: {model_name}',
                'available_models': hyperparameter_manager.get_available_models()
            }), 400
        
        # Get JSON data from request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        new_params = request.get_json()
        if not isinstance(new_params, dict):
            return jsonify({'error': 'Hyperparameters must be a dictionary'}), 400
        
        # Update hyperparameters
        updated_params = hyperparameter_manager.update_hyperparameters(model_name, new_params)
        
        return jsonify({
            'message': f'Hyperparameters updated for {model_name}',
            'model_name': model_name,
            'updated_hyperparameters': updated_params
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update hyperparameters for {model_name}: {str(e)}'}), 500

@app.route('/hyperparameters/<model_name>/reset', methods=['POST'])
def reset_hyperparameters(model_name):
    """Reset hyperparameters for a specific model to defaults."""
    try:
        if model_name not in hyperparameter_manager.get_available_models():
            return jsonify({
                'error': f'Unknown model: {model_name}',
                'available_models': hyperparameter_manager.get_available_models()
            }), 400
        
        reset_params = hyperparameter_manager.reset_hyperparameters(model_name)
        
        return jsonify({
            'message': f'Hyperparameters reset to defaults for {model_name}',
            'model_name': model_name,
            'reset_hyperparameters': reset_params
        })
    except Exception as e:
        return jsonify({'error': f'Failed to reset hyperparameters for {model_name}: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Exoplanet Detection API',
        'version': '1.0.0',
        'endpoints': {
            'POST /look-for-exoplanet': 'Detect exoplanets from light curve data',
            'GET /hyperparameters': 'Get all hyperparameters',
            'GET /hyperparameters/<model>': 'Get hyperparameters for specific model',
            'POST /hyperparameters/<model>': 'Update hyperparameters for specific model',
            'POST /hyperparameters/<model>/reset': 'Reset hyperparameters to defaults',
            'GET /health': 'Health check'
        },
        'parameters': {
            'label': 'Target name or coordinates (optional)',
            'lightcurve': 'CSV file with light curve data (optional)'
        },
        'available_models': hyperparameter_manager.get_available_models()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
