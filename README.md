# 🧠 Brain Tumor Detection API

A Flask-based REST API that powers the Brain Tumor Detection platform by providing secure user authentication, MRI image analysis, prediction history management, and PostgreSQL database integration. The API uses a TensorFlow Lite deep learning model to classify brain tumors from MRI scans and generate explainable prediction results.

---

## 🔗 Project Ecosystem

🌐 **Live Backend API**  
https://brain-tumor-api-zg3b.onrender.com

📱 **Flutter Application Repository**  
https://github.com/shalemraju1/brain-tumor-flutter-app

---

## ✨ Features

- 🔐 Secure user registration and login
- 🧠 Brain tumor classification using TensorFlow Lite
- 📤 MRI image upload and preprocessing
- 📊 Confidence score with risk level prediction
- 🔥 Heatmap generation for model explainability
- 📁 Prediction history for authenticated users
- 🗄️ PostgreSQL database integration
- 🌐 RESTful JSON API
- ☁️ Cloud deployment on Render

---

## 🩺 Tumor Classes

The model classifies MRI scans into one of the following categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Backend | Flask |
| Language | Python |
| AI Model | TensorFlow Lite |
| Image Processing | OpenCV |
| Database | PostgreSQL (Render) |
| Authentication | Session-Based Authentication |
| Deployment | Render |

---

# 📡 API Endpoints

## Authentication

### Register

```http
POST /api/register
```

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123"
}
```

---

### Login

```http
POST /api/login
```

```json
{
  "email": "john@example.com",
  "password": "password123"
}
```

---

## Prediction

```http
POST /api/predict
```

### Form Data

- image
- user_id

### Response

- Predicted Tumor Type
- Confidence Score
- Risk Level
- Heatmap Image

---

## Prediction History

```http
GET /api/history?user_id={id}
```

Returns all previous prediction reports for the authenticated user.

---

# 🗄️ Database Schema

## Users

| Field | Type |
|--------|------|
| id | Integer |
| name | Text |
| email | Unique Text |
| password | Text |

### Reports

| Field | Type |
|--------|------|
| id | Integer |
| user_id | Integer |
| prediction | Text |
| confidence | Float |
| risk_level | Text |
| created_at | Timestamp |

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/shalemraju1/braintumorclassification.git
cd backend
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file inside the backend directory.

```env
DATABASE_URL=your_postgresql_connection_url
SECRET_KEY=your_secret_key
PORT=10000
```

## Run the Server

```bash
python app.py
```

Server runs on:

```
http://127.0.0.1:10000
```

## Initialize Database

Open the following endpoints once:

```
/api/init-users
```

```
/api/init-db
```

---

# 📂 Project Structure

```
backend/
│
├── helpers/
├── routes/
├── model/
├── static/
├── templates/
├── app.py
├── requirements.txt
└── README.md
```

---

# ☁️ Deployment

**Live Backend API**

https://brain-tumor-api-zg3b.onrender.com

Hosted on **Render** with **PostgreSQL** for secure cloud-based data storage.

---

# 🔮 Future Enhancements

- JWT Authentication
- Role-Based Access Control
- Docker Support
- Batch MRI Prediction
- Swagger / OpenAPI Documentation
- Model Versioning
- Performance Monitoring & Logging

---

# 👨‍💻 Author

**Shalem Raju Bejawada**

- GitHub: https://github.com/shalemraju1
- LinkedIn: https://www.linkedin.com/in/shalem-raju-bejawada-170b40290/

Developed as part of a final-year AI/ML project focused on real-time brain tumor detection using Deep Learning, Flask, and TensorFlow Lite.