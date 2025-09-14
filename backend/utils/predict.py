# backend/utils/predict.py
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

def predict_calories_burned_simple(workout_type, difficulty, duration_minutes, age, weight, height, heart_rate):
    """
    Enhanced prediction function with improved calibration.

    🔧 Change: model paths now resolve relative to this file:
        backend/utils/models/{fitness_model.pkl, feature_names.json, label_encoders.pkl}
    """
    # --- where models live (next to this file) ---
    base_dir = Path(__file__).resolve().parent          # backend/utils
    models_dir = base_dir / 'models'                    # backend/utils/models

    model_path = models_dir / 'fitness_model.pkl'
    feature_names_path = models_dir / 'feature_names.json'
    label_encoders_path = models_dir / 'label_encoders.pkl'

    # Load model + metadata
    model = joblib.load(model_path)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    label_encoders = joblib.load(label_encoders_path)

    # Reasonable per-minute estimates to construct features
    steps_estimates = {'yoga': 25, 'strength': 30, 'cardio': 150, 'hiit': 180}
    distance_estimates = {'yoga': 0.05, 'strength': 0.06, 'cardio': 0.18, 'hiit': 0.15}

    active_minutes = duration_minutes

    input_data = {
        'age': age,
        'weight': weight,
        'height': height,
        'active_minutes': active_minutes,
        'heart_rate_avg': heart_rate,
        'steps': steps_estimates.get(workout_type, 50) * duration_minutes,
        'distance_km': distance_estimates.get(workout_type, 0.05) * duration_minutes,
        'workout_duration': duration_minutes,
        'workout_type': workout_type,
        'difficulty': difficulty
    }

    input_df = pd.DataFrame([input_data])

    # Encode categoricals with saved label encoders
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            input_df[col] = encoder.transform(input_df[col])

    # Ensure all expected features exist and are ordered
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[feature_names]

    # Predict calories per minute
    base_cpm = float(model.predict(input_df)[0])

    # Calibrate
    calibrated_cpm = apply_calibration(
        base_cpm, workout_type, difficulty, heart_rate, weight, age
    )

    # Total calories
    total_calories = calibrated_cpm * duration_minutes
    return float(total_calories)

def apply_calibration(base_cpm, workout_type, difficulty, heart_rate, weight, age):
    """
    Improved calibration based on exercise physiology
    """
    calibrated_cpm = base_cpm

    # 1) Base upward adjustment
    calibrated_cpm += 2.5

    # 2) Type multiplier (approx MET mapping)
    type_multipliers = {'yoga': 1.8, 'strength': 2.2, 'cardio': 2.8, 'hiit': 3.5}
    calibrated_cpm *= type_multipliers.get(workout_type, 2.0)

    # 3) Difficulty multiplier
    difficulty_multipliers = {'easy': 0.7, 'medium': 1.0, 'hard': 1.4}
    calibrated_cpm *= difficulty_multipliers.get(difficulty, 1.0)

    # 4) Heart-rate intensity factor
    hr_intensity = max(0.6, min(2.0, (heart_rate - 60) / 40.0))
    calibrated_cpm *= hr_intensity

    # 5) Weight factor
    calibrated_cpm *= (weight / 65.0)

    # 6) Age factor (↓ after ~25)
    age_factor = max(0.7, 1.0 - (age - 25) * 0.005)
    calibrated_cpm *= age_factor

    # 7) Realistic bounds
    realistic_min = {'yoga': 3.0, 'strength': 4.0, 'cardio': 6.0, 'hiit': 8.0}
    realistic_max = {'yoga': 8.0, 'strength': 12.0, 'cardio': 18.0, 'hiit': 22.0}
    calibrated_cpm = max(
        realistic_min.get(workout_type, 4.0),
        min(realistic_max.get(workout_type, 15.0), calibrated_cpm)
    )
    return float(calibrated_cpm)

# Optional: quick sanity test
if __name__ == "__main__":
    total = predict_calories_burned_simple(
        workout_type="strength", difficulty="medium",
        duration_minutes=30, age=30, weight=75, height=175, heart_rate=145
    )
    print("Example total calories:", round(total, 1))
