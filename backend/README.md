# Brain Tumor Detection API

## Overview

This project provides a Flask-based REST API for detecting brain tumors from MRI images using a trained deep learning model (TensorFlow Lite). The API supports user authentication, image prediction, and history tracking.

## Features

* User registration and login
* MRI image upload and tumor prediction
* Tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
* Confidence score and risk level
* Heatmap generation for explainability
* Prediction history storage using PostgreSQL

## Technology Stack

* Python (Flask)
* PostgreSQL (Render)
* TensorFlow Lite
* OpenCV

## API Endpoints

### Register

POST /api/register
Request Body:
{
"name": "string",
"email": "string",
"password": "string"
}

### Login

POST /api/login
Request Body:
{
"email": "string",
"password": "string"
}

### Predict

POST /api/predict
Form Data:

* image (file)
* user_id (integer)

### History

GET /api/history?user_id=<id>

## Database Schema

### Users Table

* id (Primary Key)
* name (Text)
* email (Unique)
* password (Text)

### Reports Table

* id (Primary Key)
* user_id (Integer)
* prediction (Text)
* confidence (Float)
* risk_level (Text)
* created_at (Timestamp)

## Setup Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

3. Initialize database:
   Visit /api/init-users and /api/init-db

## Deployment

The backend is deployed on Render.

## Notes

* Ensure PostgreSQL connection string is correctly configured.
* All endpoints return JSON responses.
* Proper error handling is implemented for production stability.

## Author

C5 Team
