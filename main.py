import os
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
from io import StringIO, BytesIO
import json

# Create FastAPI application
app = FastAPI(
    title="XGBoost Regression Model API",
    description="A user-friendly API for making predictions using a trained XGBoost model",
    version="1.0.0"
)

# Configure CORS to make API accessible publicly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

PORT = int(os.getenv("PORT", 8000))

# Create directory for model storage if it doesn't exist
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/trained_model.sav"

# Load the model if it exists, otherwise provide informative error
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    feature_names = model.feature_names_in_
    print(f"Model loaded successfully with features: {feature_names}")
except (FileNotFoundError, EOFError):
    model = None
    feature_names = []
    print("No model found. You'll need to upload a model first.")

# Set up templates and static files for the web interface
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create HTML template for the web interface
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XGBoost Regression Model API</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .card {
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card-header {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .main-container {
            max-width: 1200px;
        }
        .feature-input {
            margin-bottom: 10px;
        }
        #results {
            display: none;
            margin-top: 20px;
        }
        .prediction-value {
            font-size: 24px;
            font-weight: bold;
            color: #0d6efd;
        }
        #modelInfo {
            margin-top: 20px;
        }
        #loadingSpinner {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container main-container">
        <h1 class="mb-4 text-center">XGBoost Regression Model API</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Model Management</div>
                    <div class="card-body">
                        <div id="modelInfo" class="mb-3">
                            <h5>Current Model Status:</h5>
                            <div id="modelStatus">Checking status...</div>
                        </div>
                        
                        <form id="uploadModelForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="modelFile" class="form-label">Upload Model File (.sav)</label>
                                <input type="file" class="form-control" id="modelFile" name="model_file" accept=".sav,.pkl,.pickle">
                            </div>
                            <button type="submit" class="btn btn-primary">Upload Model</button>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">Batch Predictions</div>
                    <div class="card-body">
                        <form id="uploadCsvForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="csvFile" class="form-label">Upload CSV File</label>
                                <input type="file" class="form-control" id="csvFile" name="file" accept=".csv">
                                <div class="form-text">CSV should contain all required feature columns.</div>
                            </div>
                            <button type="submit" class="btn btn-primary">Get Batch Predictions</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Single Prediction</div>
                    <div class="card-body">
                        <form id="singlePredictionForm">
                            <div id="featureInputs">
                                <div class="alert alert-info">
                                    Please upload a model first to see the required features.
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary" id="predictBtn">Get Prediction</button>
                            <div id="loadingSpinner" class="spinner-border text-primary mt-3" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </form>
                        
                        <div id="results" class="mt-4">
                            <h5>Prediction Result:</h5>
                            <div class="prediction-value" id="predictionValue"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">API Documentation</div>
            <div class="card-body">
                <h5>Endpoints:</h5>
                <ul>
                    <li><code>GET /api/model/info</code> - Get information about the current model</li>
                    <li><code>POST /api/model/upload</code> - Upload a new model</li>
                    <li><code>POST /api/predict</code> - Make a single prediction</li>
                    <li><code>POST /api/predict/batch</code> - Make batch predictions from CSV</li>
                </ul>
                <p>For more details, visit <a href="/docs" target="_blank">API Documentation</a></p>
            </div>
        </div>
    </div>

    <script>
        // Function to check model status
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/model/info');
                const data = await response.json();
                
                const statusDiv = document.getElementById('modelStatus');
                
                if (data.status === 'loaded') {
                    statusDiv.innerHTML = `
                        <div class="alert alert-success">
                            Model loaded successfully!
                            <hr>
                            <p><strong>Features:</strong> ${data.features.join(', ')}</p>
                        </div>
                    `;
                    
                    // Generate feature input fields
                    const featureInputs = document.getElementById('featureInputs');
                    featureInputs.innerHTML = '';
                    
                    data.features.forEach(feature => {
                        const div = document.createElement('div');
                        div.className = 'mb-3 feature-input';
                        div.innerHTML = `
                            <label for="${feature}" class="form-label">${feature}</label>
                            <input type="number" step="any" class="form-control" id="${feature}" name="${feature}" required>
                        `;
                        featureInputs.appendChild(div);
                    });
                } else {
                    statusDiv.innerHTML = `
                        <div class="alert alert-warning">
                            No model loaded. Please upload a model file.
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error checking model status:', error);
                document.getElementById('modelStatus').innerHTML = `
                    <div class="alert alert-danger">
                        Error checking model status. Please try refreshing the page.
                    </div>
                `;
            }
        }
        
        // Handle model upload
        document.getElementById('uploadModelForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('modelFile');
            
            if (fileInput.files.length === 0) {
                alert('Please select a model file');
                return;
            }
            
            formData.append('model_file', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/model/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    alert('Model uploaded successfully!');
                    checkModelStatus();
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                console.error('Error uploading model:', error);
                alert('Error uploading model. Please try again.');
            }
        });
        
        // Handle single prediction
        document.getElementById('singlePredictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const loadingSpinner = document.getElementById('loadingSpinner');
            loadingSpinner.style.display = 'inline-block';
            
            const formData = {};
            const inputs = document.querySelectorAll('#featureInputs input');
            
            inputs.forEach(input => {
                formData[input.name] = parseFloat(input.value);
            });
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('predictionValue').textContent = result.prediction.toFixed(4);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                console.error('Error getting prediction:', error);
                alert('Error getting prediction. Please try again.');
            } finally {
                loadingSpinner.style.display = 'none';
            }
        });
        
        // Handle CSV upload for batch predictions
        document.getElementById('uploadCsvForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('csvFile');
            
            if (fileInput.files.length === 0) {
                alert('Please select a CSV file');
                return;
            }
            
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/predict/batch', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    // Trigger download of the predictions CSV
                    const blob = await response.blob();
                    const downloadUrl = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = downloadUrl;
                    a.download = 'predictions.csv';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    alert('Batch predictions completed! Downloading results...');
                } else {
                    const result = await response.json();
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                console.error('Error getting batch predictions:', error);
                alert('Error getting batch predictions. Please try again.');
            }
        });
        
        // Check model status on page load
        window.addEventListener('DOMContentLoaded', checkModelStatus);
    </script>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """)

# Create CSS file for the web interface
with open("static/style.css", "w") as f:
    f.write("""
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h1 {
    color: #0066cc;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    margin-bottom: 20px;
    border: none;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.card-header {
    background-color: #0066cc;
    color: white;
    font-weight: bold;
    padding: 15px;
}

.card-body {
    padding: 20px;
}

.btn-primary {
    background-color: #0066cc;
    border-color: #0066cc;
}

.btn-primary:hover {
    background-color: #0055aa;
    border-color: #0055aa;
}

.form-control:focus {
    border-color: #0066cc;
    box-shadow: 0 0 0 0.25rem rgba(0, 102, 204, 0.25);
}
    """)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define data models for API
class PredictionInput(BaseModel):
    """Dynamic model for prediction input - will be populated with feature names from the model"""
    pass

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted value from the model")

# Dynamic route to serve the web interface
@app.get("/", response_class=JSONResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API routes for model management and predictions
@app.get("/api/model/info", response_class=JSONResponse)
async def model_info():
    """Get information about the current model"""
    if model is not None:
        return {
            "status": "loaded",
            "features": list(feature_names)
        }
    else:
        return {
            "status": "not_loaded",
            "features": []
        }

@app.post("/api/model/upload", response_class=JSONResponse)
async def upload_model(model_file: UploadFile = File(...)):
    """Upload a new model file (.sav, .pkl, or .pickle)"""
    global model, feature_names
    
    # Validate file type
    if not model_file.filename.endswith(('.sav', '.pkl', '.pickle')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .sav, .pkl, or .pickle file")
    
    try:
        # Save the uploaded model
        content = await model_file.read()
        with open(MODEL_PATH, 'wb') as f:
            f.write(content)
        
        # Load the model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        # Extract feature names
        feature_names = model.feature_names_in_
        
        return {
            "status": "success",
            "message": "Model uploaded and loaded successfully",
            "features": list(feature_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/api/predict", response_class=JSONResponse)
async def predict(data: Dict[str, float]):
    """Make a single prediction using the loaded model"""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please upload a model first.")
    
    try:
        # Convert input data to DataFrame with correct feature order
        input_df = pd.DataFrame([data])
        
        # Check if all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {', '.join(missing_features)}"
            )
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return {"prediction": float(prediction)}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from a CSV file"""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please upload a model first.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    try:
        # Read CSV content
        content = await file.read()
        input_df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Check if all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features in CSV: {', '.join(missing_features)}"
            )
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Add predictions to the dataframe
        input_df['prediction'] = predictions
        
        # Return CSV with predictions
        output = StringIO()
        input_df.to_csv(output, index=False)
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Helper function to create a sample model for testing
def create_sample_model():
    """Create a sample model for testing if no model exists"""
    from sklearn.datasets import make_regression
    from xgboost import XGBRegressor
    
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    
    # Create feature names
    feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_cols)
    
    # Train a simple model
    sample_model = XGBRegressor(n_estimators=10)
    sample_model.fit(X_df, y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(sample_model, f)
    
    print("Sample model created for testing")
    return sample_model, feature_cols

# Main entry point
if __name__ == "__main__":
    # Create a sample model if no model exists
    if model is None:
        print("No model found. Creating a sample model for testing...")
        model, feature_names = create_sample_model()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # For deployment environments that import this file
    if model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        feature_names = model.feature_names_in_
