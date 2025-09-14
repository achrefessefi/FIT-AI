# 🏋️ FIT-AI: AI-Powered Fitness Coach

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
├── backend/                 # FastAPI backend server
│   ├── app/
│   │   ├── main.py         # Main FastAPI application
│   │   └── core/
│   │       └── config.py   # Configuration settings
│   ├── data/               # JSON data storage
│   └── utils/
│       └── predict.py      # ML prediction utilities
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   └── api.js          # API communication
│   └── package.json
├── push.py                 # Push-up analysis module
├── squat.py                # Squat analysis module
└── yoga.py                 # Yoga pose detection module
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
   uvicorn main:app --reload --port 8000
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

### Backend
- **FastAPI**: High-performance web framework
- **OpenCV**: Computer vision for exercise tracking
- **MediaPipe**: Pose estimation and landmark detection
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models
- **GROQ**: AI coaching integration (optional)

### Frontend
- **React 18**: Modern UI framework
- **Vite**: Fast build tool and dev server
- **Chart.js**: Data visualization
- **React Router**: Navigation
- **Lucide React**: Icon library

### Computer Vision
- **MediaPipe Pose**: Real-time pose estimation
- **OpenCV**: Image processing and camera handling
- **Custom Algorithms**: Exercise-specific form analysis

## 📊 Features Overview

### Exercise Tracking
- **Real-time pose detection** using MediaPipe
- **Form analysis** with angle calculations and alignment checks
- **Rep counting** with automatic detection
- **Performance metrics** including speed, accuracy, and consistency

### AI Coaching
- **Voice feedback** during workouts
- **Form corrections** with specific guidance
- **Personalized recommendations** based on performance
- **Progress tracking** with detailed analytics

### Dashboard Analytics
- **Weekly progress visualization**
- **Calorie tracking and comparison**
- **Performance charts** with historical data
- **Workout scheduling** and goal setting

## 🎯 Exercise Modules

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

## 🔮 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /predict-calories` - Calorie burn prediction
- `GET /dashboard-data` - Dashboard analytics
- `POST /save-workout` - Workout data storage

### Exercise Integration
- Real-time form analysis
- Performance metrics calculation
- Progress tracking updates
- AI coaching feedback

## 🛠️ Development

### Project Structure
```
├── backend/
│   ├── app/main.py          # FastAPI routes and middleware
│   ├── core/config.py       # Application configuration
│   ├── data/               # JSON data storage
│   └── utils/predict.py    # ML prediction models
├── frontend/
│   ├── src/components/     # React UI components
│   ├── src/api.js         # Backend API communication
│   └── package.json       # Dependencies and scripts
└── *.py                   # Exercise tracking modules
```

### Running in Development Mode

1. **Backend**: `uvicorn main:app --reload`
2. **Frontend**: `npm run dev`
3. **Exercise Modules**: `python [module_name].py`

### Building for Production

**Frontend**:
```bash
npm run build
npm run preview
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the code examples in each module

## 🎉 Acknowledgments

- **MediaPipe** for pose estimation technology
- **OpenCV** for computer vision capabilities
- **React** and **FastAPI** communities for excellent frameworks
- **Chart.js** for beautiful data visualizations

---

**Ready to transform your fitness journey with AI? Get started now!** 🚀💪
