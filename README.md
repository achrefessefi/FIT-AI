# 🏋️ FIT-UP: AI-Powered Fitness Coach

A comprehensive AI-powered fitness application that provides real-time workout analysis, form correction, and personalized coaching through computer vision and machine learning.

## ✨ Features

### 🎯 Real-Time Exercise Analysis
- **Push-up Tracking**: Real-time form analysis with posture correction
- **Squat Monitoring**: Depth and alignment feedback with rep counting
- **Yoga Pose Detection**: Pose validation and guidance

### 📊 Smart Analytics
- **Performance Tracking**: Detailed metrics and progress visualization
- **Calorie Prediction**: ML-powered calorie burn estimation
- **Weekly Progress Reports**: Comprehensive fitness journey tracking
- **Motivation Score**: Session-level score combining pace consistency, rest discipline, and rep quality

### 🎶 Mood-Adaptive Soundtrack
- **Emotion Detection (DeepFace)**: Live facial analysis (happy/neutral/sad/angry/surprise/fear) fused with the **motivation score**
- **Auto Music Switching**: Dynamically changes **playlist/tempo/energy** to boost low motivation or calm post-set recovery
- **Context Fusion**: Music decision = f(**emotion**, **motivation**, **weather**, **time of day**)
  - Low motivation + evening → energetic tracks to re-engage
  - High intensity set complete → downshift to calmer tracks during rest
  - Hot weather → lighter tempo to control heart rate drift
- **Seamless Transitions**: Crossfade and latency-aware switching so audio never feels jarring

### 🤖 AI Coach
- **Form Feedback**: Real-time corrections and suggestions
- **Voice Guidance**: Audio coaching during workouts
- **Personalized Recommendations**: Tailored workout plans

### 💻 Modern Dashboard
- **Interactive Charts**: Performance visualization with Chart.js
- **Progress Tracking**: Weekly goals and achievement monitoring
- **Workout Scheduling**: Plan and track your fitness routine

## 🏗️ Architecture

```
FIT-AI/
├── backend/                          # FastAPI backend server
│   ├── app/
│   │   ├── main.py                   # FastAPI app entrypoint
│   │   ├── core/
│   │   │   └── config.py             # Settings / env loader
│   │   ├── utils/
│   │   │   ├── predict.py            # ML prediction helpers
│   │   │   └── models/               # Saved models / artifacts
│   │   │       └── fitness_model.pkl # Example model file
│   │   ├── routers/                  # (optional) API route modules
│   │   └── schemas/                  # (optional) Pydantic models
│   ├── data/                         # JSON/data storage (if needed)
│   ├── requirements.txt              # Backend Python deps
│   └── .env.example                  # Example env vars (no secrets)
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── components/               # React components
│   │   └── api.js                    # API client
│   ├── package.json
│   └── .env.example                  # Example frontend envs
│
├── push.py                           # Push-up analysis script
├── squat.py                          # Squat analysis script
├── mood_music.py                     # Mood analysis + music switching
└── yoga.py                           # Yoga pose detection script

```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam for exercise tracking
- (Optional) GROQ API key for AI coaching features

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (optional):
   ```bash
   # Create .env file in project root
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-70b-versatile
   ```

4. **Start the backend server**:
   ```bash
   cd app
   uvicorn app.main:app --reload --app-dir backend
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Access the application**:
   Open [http://localhost:5173](http://localhost:5173) in your browser

### Exercise Modules

Run individual exercise tracking modules:

```bash
# Push-up tracking
python push.py

# Squat analysis
python squat.py

# Yoga pose detection
python yoga.py
```

## 🔧 Technology Stack

**3 AI Types Fusion:** Computer Vision (CV) + Generative AI (GEN) + Predictive AI (PRED)

### Backend
- **FastAPI** – high-performance API server
- **OpenCV** – video capture & image processing
- **MediaPipe BlazePose** – real-time body landmarks for form tracking
- **DeepFace** – on-camera emotion detection (mood → music switching)
- **NumPy** – numerical ops
- **Scikit-learn (Random Forest + evaluation)** – calorie prediction & scoring models
- *(Optional)* **GROQ (LLaMA-3.1-70B)** – Gen-AI coaching/tips generation

### Frontend
- **React 18** – modern UI framework
- **Vite** – ultra-fast dev/build tooling
- **Chart.js** – performance & mood visualizations
- **React Router** – navigation
- **Lucide React** – icon set

### Computer Vision
- **MediaPipe Pose (BlazePose)** – pose estimation
- **OpenCV** – preprocessing, camera handling, overlays
- **DeepFace** – emotion classification (happy/neutral/sad/angry/…)
- **Custom Algorithms** – exercise-specific angle, cadence, depth & rep logic

### Generative AI
- **GROQ (LLaMA-3.1/70B)** – natural-language coaching, adaptive cues, summaries

### Predictive AI
- **Scikit-learn: Random Forest** – calorie burn & motivation scoring
- **Model evaluation** – metrics tracking and iteration

### Data Handling & Integrations
- **Structured JSON** – session logs, configs, analytics
- **CSV** – performance summaries
- **OpenWeatherMap** – weather-aware coaching & soundtrack adjustments


### 🗂️ Data

- **Fitness Dataset (Hackathon JSON format):**
  - We used the official hackathon-provided fitness data in **JSON** format (sessions, reps, timestamps, pose metrics).
  - Stored under `backend/data/` and consumed by the FastAPI endpoints and analysis scripts.


- **Weather Data (API):**
  - Retrieved via a **Weather API** (e.g., OpenWeatherMap/WeatherAPI) to adapt coaching + music to ambient conditions.
 

## 🎯 Exercise Modules (Prototype)

### Push-up Analysis (`push.py`)
- Elbow angle tracking for proper form
- Hip alignment monitoring
- Rep counting with form validation
- Real-time feedback on posture

### Squat Monitoring (`squat.py`)
- Knee angle analysis for proper depth
- Back alignment checking
- Balance and stability assessment
- Progressive difficulty tracking

### Yoga Pose Detection (`yoga.py`)
- Multiple pose recognition
- Alignment guidance
- Hold duration tracking
- Breathing pattern analysis


