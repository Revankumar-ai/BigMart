# BigMart
XGBoost Model API
A user-friendly web API for deploying XGBoost regression models with public access. This solution allows users to upload trained models, make single predictions through an intuitive interface, and process batch predictions with CSV files.
Features

Web-based Interface: Clean, responsive UI for all model interactions
Model Management: Upload and manage XGBoost models (.sav format)
Single Predictions: Make predictions one at a time with a form interface
Batch Predictions: Upload CSV files with multiple samples for bulk predictions
REST API: Well-documented API endpoints for programmatic access
Responsive Design: Works on desktop and mobile devices
Interactive Documentation: Automatic API documentation with Swagger UI

Screenshots
Show Image
Quick Start

Install the required dependencies:
pip install -r requirements.txt

Run the server:
python main.py

Open your browser and go to http://localhost:8000
Upload your trained XGBoost model (.sav file)
Start making predictions!

API Endpoints

GET /api/model/info - Get information about the current model
POST /api/model/upload - Upload a new model file
POST /api/predict - Make a single prediction
POST /api/predict/batch - Make batch predictions from CSV

Full documentation available at /docs when the server is running.
Requirements

Python 3.8+
FastAPI
XGBoost
Pandas
scikit-learn
uvicorn

See requirements.txt for the complete list.
Use Cases

Data scientists who want to quickly deploy their models
Organizations that need a simple prediction API
Teams that want to provide ML model access to non-technical users
Educational environments for demonstrating ML model deployment

Deployment
See the Deployment Guide for detailed instructions on deploying to:

Local servers
Cloud platforms (AWS, Azure, GCP)
Docker containers
Heroku

Development
Project Structure
├── main.py           # Main application file
├── requirements.txt  # Python dependencies
├── models/           # Directory for storing models
├── static/           # Static files (CSS, JS)
└── templates/        # HTML templates
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

FastAPI
XGBoost
Bootstrap
