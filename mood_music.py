# mood_music_coach.py
import cv2
import time
import requests
import os
import random
from datetime import datetime
from deepface import DeepFace
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import pygame, but make it optional
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Pygame not available - music playback will be simulated")


class EmotionDetector:
    """Detects emotions using DeepFace"""
    
    def __init__(self):
        self.detection_active = False
        self.detection_start_time = None
        self.detection_duration = 5.0
        
    def start_detection(self):
        """Start 5-second emotion detection"""
        self.detection_active = True
        self.detection_start_time = time.time()
        print("🎯 Starting 5-second emotion detection...")
        
    def detect_emotion_from_frame(self, frame) -> Optional[str]:
        """Detect emotion using DeepFace"""
        if not self.detection_active:
            return None
            
        # Check if 5 seconds passed
        if time.time() - self.detection_start_time > self.detection_duration:
            self.detection_active = False
            emotion = self._analyze_emotion(frame)
            print(f"😊 Emotion detection complete! Detected: {emotion}")
            return emotion
            
        return None
        
    def _analyze_emotion(self, frame) -> str:
        """Analyze emotion using DeepFace"""
        try:
            # Use DeepFace to analyze emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Get dominant emotion
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']
                
            dominant_emotion = max(emotions, key=emotions.get)
            
            # Map DeepFace emotions to fitness categories
            emotion_mapping = {
                'happy': 'energetic',
                'surprise': 'energetic', 
                'neutral': 'focus',
                'sad': 'recovery',
                'angry': 'intense',
                'disgust': 'intense',
                'fear': 'motivational'
            }
            
            fitness_mood = emotion_mapping.get(dominant_emotion, 'focus')
            print(f"🔍 DeepFace emotion: {dominant_emotion} → Fitness mood: {fitness_mood}")
            
            return fitness_mood
            
        except Exception as e:
            print(f"❌ Emotion detection failed: {e}")
            return 'focus'  # Default fallback


class WeatherService:
    """Simple weather service"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    def get_weather(self) -> str:
        """Get current weather"""
        if not self.api_key:
            weather_options = ["sunny", "cloudy", "rainy", "stormy"]
            return random.choice(weather_options)
            
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={self.api_key}"
            response = requests.get(url, timeout=5)
            data = response.json()
            weather_main = data["weather"][0]["main"].lower()
            
            weather_mapping = {
                'clear': 'sunny',
                'clouds': 'cloudy', 
                'rain': 'rainy',
                'thunderstorm': 'stormy'
            }
            
            return weather_mapping.get(weather_main, 'cloudy')
            
        except Exception:
            return 'cloudy'


class TimeAnalyzer:
    """Simple time analysis"""
    
    @staticmethod
    def get_time_period() -> str:
        """Get current time period"""
        hour = datetime.now().hour
        return "day" if 6 <= hour < 18 else "night"


class MusicPlayer:
    """Simple music player"""
    
    def __init__(self, music_directory: str = "./music"):
        self.music_directory = music_directory
        
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            
        # Clean music mapping
        self.music_mapping = {
            "energetic": {
                "day": ["high_energy_workout.mp3", "power_boost.mp3"],
                "night": ["evening_power.mp3", "night_energy.mp3"]
            },
            "motivational": {
                "day": ["push_through.mp3", "determination.mp3"],
                "night": ["night_motivation.mp3", "never_quit.mp3"]
            },
            "intense": {
                "day": ["intense_training.mp3", "beast_mode.mp3"],
                "night": ["midnight_grind.mp3", "hardcore_night.mp3"]
            },
            "recovery": {
                "day": ["gentle_recovery.mp3", "calm_restore.mp3"],
                "night": ["rest_mode.mp3", "rest_mode.mp3"]
            },
            "focus": {
                "day": ["focused_flow.mp3", "steady_concentration.mp3"],
                "night": ["night_focus.mp3", "night_focus.mp3"]
            }
        }
        
    def select_and_play(self, emotion: str, weather: str, time_period: str):
        """Select and play music based on inputs"""
        print(f"🎵 Selection: Emotion={emotion}, Weather={weather}, Time={time_period}")
        
        # Weather influence
        if weather == "rainy" and emotion == "energetic":
            emotion = "intense"  # Indoor intense workout
        elif weather == "sunny" and emotion == "recovery":
            emotion = "motivational"  # Sunshine boost
            
        # Get tracks
        tracks = self.music_mapping[emotion][time_period]
        selected_track = random.choice(tracks)
        
        print(f"🎼 Playing: {selected_track}")
        self._play_track(selected_track)
        
    def _play_track(self, track_name: str):
        """Play the track"""
        track_path = os.path.join(self.music_directory, track_name)
        
        if not os.path.exists(track_path):
            print(f"🎶 [DEMO] Would play: {track_name}")
            return
            
        if PYGAME_AVAILABLE:
            pygame.mixer.music.load(track_path)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            print(f"🎶 Now playing: {track_name}")
        else:
            print(f"🎶 [DEMO] Would play: {track_name}")


class FitnessMusicCoach:
    """Main fitness music coach - clean and simple"""
    
    def __init__(self, weather_api_key: Optional[str] = None):
        self.cap = cv2.VideoCapture(0)
        self.emotion_detector = EmotionDetector()
        self.weather_service = WeatherService(weather_api_key)
        self.time_analyzer = TimeAnalyzer()
        self.music_player = MusicPlayer()
        self.music_selected = False
        
    def run(self):
        """Main run loop"""
        print("🚀 Starting Fitness Music Coach...")
        print("📷 Look at camera for emotion detection")
        print("Press 'q' to quit, 'r' to restart detection\n")
        
        self.emotion_detector.start_detection()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Show frame with status
            self._draw_status(frame)
            cv2.imshow('Fitness Music Coach', frame)
            
            # Detect emotion
            if not self.music_selected:
                detected_emotion = self.emotion_detector.detect_emotion_from_frame(frame)
                if detected_emotion:
                    self._select_music(detected_emotion)
                    self.music_selected = True
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._restart_detection()
                
        self._cleanup()
        
    def _draw_status(self, frame):
        """Draw status on frame"""
        if self.music_selected:
            cv2.putText(frame, "Music Playing! Press 'r' to restart", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Show your emotion for music selection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
    def _select_music(self, emotion: str):
        """Select music based on emotion, weather, and time"""
        weather = self.weather_service.get_weather()
        time_period = self.time_analyzer.get_time_period()
        
        print(f"\n🎯 Analysis: Emotion={emotion}, Weather={weather}, Time={time_period}")
        self.music_player.select_and_play(emotion, weather, time_period)
        
    def _restart_detection(self):
        """Restart detection"""
        print("\n🔄 Restarting detection...")
        self.music_selected = False
        self.emotion_detector.start_detection()
        
    def _cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()


if __name__ == "__main__":
    # Load weather API key from .env file
    weather_api_key = os.getenv('WEATHER_API_KEY')
    
    if not weather_api_key:
        print("⚠️  No WEATHER_API_KEY found in .env file - using demo mode")
    
    coach = FitnessMusicCoach(weather_api_key)
    coach.run()