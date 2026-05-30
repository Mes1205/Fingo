# 🔌 Fingo API Integration Guide

**Panduan lengkap untuk mengintegrasikan Fingo AI API ke backend aplikasi kamu.**

---

## 📋 Daftar Isi
1. [Overview API](#overview-api)
2. [Endpoint yang Tersedia](#endpoint-yang-tersedia)
3. [Setup Backend](#setup-backend)
4. [Implementasi di Backend](#implementasi-di-backend)
5. [Contoh Integrasi](#contoh-integrasi)
6. [Error Handling](#error-handling)
7. [Deployment](#deployment)

---

## Overview API

### Apa itu Fingo API?
Fingo adalah AI-powered financial assistant khusus untuk gig workers Indonesia. API menyediakan:
- ✅ **Prediksi Pendapatan** — Prediksi income 4 minggu ke depan
- ✅ **Chat AI** — Konsultasi keuangan dengan Gemini AI
- ✅ **Impulsive Spending Detection** — Deteksi kebiasaan belanja impulsif

### Endpoint Base
```
Staging:  http://localhost:8000
Production: https://mes1205-fingo.hf.space
```

---

## 🔌 Endpoint yang Tersedia

### 1. **POST /predict/income**
Prediksi pendapatan berdasarkan historical data.

**Request:**
```json
{
  "income_history": [5000000, 4800000, 5200000, 4900000, ...]  // 12 minggu terakhir
}
```

**Response (200 OK):**
```json
{
  "prediction_next_week": 5100000,
  "prediction_4_weeks_ahead": [5100000, 5200000, 5150000, 5300000],
  "total_projected_income": 20750000,
  "income_direction": "Up",
  "avg_income_last_4w": 5050000,
  "confidence": 0.85
}
```

**Error Responses:**
- `400 Bad Request` — Format data tidak sesuai
- `422 Unprocessable Entity` — Data validation error
- `500 Internal Server Error` — Model prediction error

---

### 2. **POST /chat**
Chat dengan Fingo AI untuk konsultasi keuangan.

**Request:**
```json
{
  "user_message": "Gimana cara manage gaji aku bulan ini?",
  "financial_context": {
    "income": 5000000,
    "expense": 3500000,
    "budget_remaining": 1500000,
    "impulsive_count": 3
  }
}
```

**Response (200 OK):**
```json
{
  "reply": "Dengan income Rp 5jt dan pengeluaran Rp 3.5jt, kamu punya sisa Rp 1.5jt. Rekomendasi saya...",
  "suggestions": ["Sisihkan 20% untuk dana darurat", "Kuratin pengeluaran impulsif"]
}
```

---

## 🛠️ Setup Backend

### Prerequisites
```bash
Python 3.9+
FastAPI
Uvicorn
TensorFlow 2.14+
pandas
scikit-learn
joblib
python-dotenv
google-generativeai  # Untuk Gemini API
```

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn==0.24.0
tensorflow==2.14.0
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.1
numpy==1.24.3
python-dotenv==1.0.0
google-generativeai==0.3.0
pydantic==2.4.2
```

### 2. Setup Environment Variables
Buat file `.env` di root project:
```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Model Paths
MODEL_DIR=./outputs/saved_model
DL_MODEL_PATH=./outputs/saved_model/fingo_dl_v6.keras
CONFIG_PATH=./outputs/saved_model/final_config.json

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### 3. Struktur Folder
```
DBS_Capstone_Project/
├── backend/
│   ├── main.py                    # FastAPI main app
│   ├── models.py                  # Pydantic models
│   ├── predictor.py               # Income prediction logic
│   ├── chat_handler.py            # Chat AI logic
│   └── config.py                  # Configuration
├── outputs/
│   └── saved_model/
│       ├── fingo_dl_v6.keras
│       ├── final_config.json
│       ├── feature_cols.json
│       └── *.joblib
├── .env
├── requirements.txt
└── fingoYuhu.py                   # Streamlit frontend
```

---

## 💻 Implementasi di Backend

### 1. **main.py** — FastAPI Server
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from predictor import IncomePredictor
from chat_handler import FingoChat

load_dotenv()

app = FastAPI(
    title="Fingo AI API",
    description="Financial Assistant for Gig Workers",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
predictor = IncomePredictor(os.getenv('MODEL_DIR'))
fingo_chat = FingoChat(os.getenv('GEMINI_API_KEY'))

# ═══════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════

class IncomeHistoryRequest(BaseModel):
    income_history: list[float]  # 12 minggu terakhir

class ChatRequest(BaseModel):
    user_message: str
    financial_context: dict

# ═══════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Fingo AI API"}

@app.post("/predict/income")
def predict_income(request: IncomeHistoryRequest):
    """
    Prediksi pendapatan 4 minggu ke depan
    """
    try:
        if len(request.income_history) < 12:
            raise HTTPException(status_code=400, detail="Minimal 12 minggu data history")
        
        prediction = predictor.predict(request.income_history)
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_fingo(request: ChatRequest):
    """
    Chat dengan AI untuk konsultasi keuangan
    """
    try:
        reply = fingo_chat.get_response(
            user_message=request.user_message,
            context=request.financial_context
        )
        return {"reply": reply}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

### 2. **predictor.py** — Income Prediction Logic
```python
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

class IncomePredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._load_models()
    
    def _load_models(self):
        """Load semua artifacts dari saved_model"""
        config_path = f"{self.model_dir}/final_config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.dl_model = keras.models.load_model(
            f"{self.model_dir}/fingo_dl_v6.keras",
            custom_objects={'ResidualBlock': ResidualBlock, 'HuberMAELoss': HuberMAELoss}
        )
        
        self.target_scaler = joblib.load(f"{self.model_dir}/local_target_scaler.joblib")
        self.feature_scaler = joblib.load(f"{self.model_dir}/feature_scaler.joblib")
        
        with open(f"{self.model_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)['feature_columns']
    
    def predict(self, income_history: list) -> dict:
        """
        Prediksi pendapatan 4 minggu ke depan
        
        Args:
            income_history: List 12 nilai pendapatan mingguan terakhir
        
        Returns:
            dict dengan prediksi dan analisis
        """
        # Normalisasi input
        income_array = np.array(income_history).reshape(-1, 1)
        income_norm = self.target_scaler.transform(income_array).flatten()
        
        # Ambil 4 nilai terakhir
        recent_income = income_norm[-4:]
        
        # Feature engineering (simplified)
        features = self._extract_features(recent_income)
        features_scaled = self.feature_scaler.transform([features])
        
        # Predict
        pred_norm, _ = self.dl_model.predict(features_scaled)
        pred_norm = np.clip(pred_norm, 0, 1)[0][0]
        
        # Inverse transform
        pred_idr = self.target_scaler.inverse_transform([[pred_norm]])[0][0]
        
        # Project 4 weeks
        predictions_4w = self._project_4_weeks(pred_idr, recent_income)
        
        # Determine direction
        avg_last_4w = np.mean(income_history[-4:])
        direction = "Up" if pred_idr > avg_last_4w else ("Down" if pred_idr < avg_last_4w * 0.95 else "Stable")
        
        return {
            "prediction_next_week": float(pred_idr),
            "prediction_4_weeks_ahead": [float(p) for p in predictions_4w],
            "total_projected_income": float(sum(predictions_4w)),
            "income_direction": direction,
            "avg_income_last_4w": float(avg_last_4w),
            "confidence": 0.85
        }
    
    def _extract_features(self, income_data):
        """Extract features dari income data"""
        return {
            'rolling_mean': np.mean(income_data),
            'rolling_std': np.std(income_data),
            'trend': income_data[-1] - income_data[0]
        }
    
    def _project_4_weeks(self, next_week_pred, recent_income, num_weeks=4):
        """Project income untuk 4 minggu ke depan"""
        trend = recent_income[-1] - recent_income[-4]
        projections = []
        
        for i in range(1, num_weeks + 1):
            proj = next_week_pred + (trend * i * 0.3)
            proj = max(proj, 0)  # Jangan negatif
            projections.append(proj)
        
        return projections
```

---

### 3. **chat_handler.py** — Gemini Chat
```python
import google.generativeai as genai

class FingoChat:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
    
    def get_response(self, user_message: str, context: dict) -> str:
        """
        Dapatkan response dari Gemini AI
        
        Args:
            user_message: Pertanyaan user
            context: Financial context (income, expense, etc.)
        
        Returns:
            Response dari AI
        """
        # Build context prompt
        income = context.get('income', 0)
        expense = context.get('expense', 0)
        budget = context.get('budget_remaining', 0)
        impulsive = context.get('impulsive_count', 0)
        
        system_prompt = f"""
Kamu adalah Fingo, financial assistant untuk gig workers Indonesia.
Konteks financial user:
- Pendapatan bulan ini: Rp {income:,.0f}
- Pengeluaran: Rp {expense:,.0f}
- Sisa budget: Rp {budget:,.0f}
- Belanja impulsif: {impulsive} kali

Berikan konsultasi keuangan yang praktis, friendly, dan relevan untuk gig workers.
Gunakan bahasa informal, hindari jargon finance yang berat.
"""
        
        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\nUser: {user_message}"
            )
            return response.text
        
        except Exception as e:
            return f"Maaf, Fingo lagi error: {str(e)}"
```

---

## 📝 Contoh Integrasi

### Backend + Frontend (Node.js/JavaScript)
```javascript
// Prediksi Income
async function predictIncome(incomeHistory) {
  const response = await fetch('http://localhost:8000/predict/income', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ income_history: incomeHistory })
  });
  return await response.json();
}

// Chat
async function chatWithFingo(message, context) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_message: message,
      financial_context: context
    })
  });
  return await response.json();
}

// Usage
const prediction = await predictIncome([5000000, 4800000, ...]);
const chat = await chatWithFingo("Gimana tips hemat kamu?", {
  income: 5000000,
  expense: 3500000,
  budget_remaining: 1500000,
  impulsive_count: 3
});
```

---

## ⚠️ Error Handling

### Common Errors & Solutions

| Error | Penyebab | Solusi |
|-------|---------|--------|
| `400 Bad Request` | Format JSON salah | Pastikan format sesuai spec |
| `422 Unprocessable Entity` | Data validation error | Cek tipe data & required fields |
| `500 Internal Server Error` | Model crash | Cek logs & file models |
| `Connection refused` | API server tidak jalan | Start server dengan `python main.py` |
| `GEMINI_API_KEY not set` | Missing env var | Tambah ke `.env` dan load dengan `dotenv` |

### Response Error Format
```json
{
  "detail": "Error message description"
}
```

---

## 🚀 Deployment

### 1. Local Testing
```bash
cd DBS_Capstone_Project
source venv311/bin/activate
python backend/main.py
```

Akses: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### 2. Deploy ke Hugging Face Spaces
```bash
# Clone Space
git clone https://huggingface.co/spaces/mes1205/fingo
cd fingo

# Copy files
cp backend/ .
cp outputs/ .
cp .env .
cp requirements.txt .

# Commit & Push
git add .
git commit -m "Deploy Fingo Backend API v1.0"
git push
```

### 3. Deploy ke Cloud (Render/Railway/Heroku)
```bash
# Buat Procfile
echo "web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT" > Procfile

# Push ke Git
git push heroku main
```

---

## 📚 API Documentation

Akses full Swagger documentation:
```
http://localhost:8000/docs
```

---

## 🎯 Summary

| Task | File | Command |
|------|------|---------|
| Run Backend | `backend/main.py` | `python backend/main.py` |
| Test API | Postman/curl | `curl http://localhost:8000/health` |
| View Docs | Swagger UI | Visit `/docs` |
| Deploy | Hugging Face/Cloud | `git push` |

---

**Questions?** Check the code comments atau baca [FastAPI docs](https://fastapi.tiangolo.com/)

Happy coding! 🚀
