# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Literal, List, Annotated
from pydantic import BaseModel, Field
from pathlib import Path
import json
import os
import subprocess, tempfile, re
from datetime import datetime
import sys  # <<< added
# at top with other imports
import time

from app.core.config import settings, get_cors_origins
from utils.predict import predict_calories_burned_simple

# Root/project paths & interpreter (ensure we run scripts with the same venv)
ROOT_DIR = Path(__file__).resolve().parents[2]   # -> project root (e.g., C:\Users\MSI\Desktop\hack)
PYTHON_EXE = sys.executable                      # -> venv python (…\venv\Scripts\python.exe)


# -------------------------------
# Groq client setup (optional)
# -------------------------------
try:
    from groq import Groq
except Exception:
    Groq = None  # library not installed

api_key = os.environ.get("GROQ_API_KEY")
cfg = {"GROQ_MODEL": os.environ.get("GROQ_MODEL")}
model = cfg.get("GROQ_MODEL", "llama-3.1-70b-versatile")

GROQ_CLIENT = None
if Groq and api_key:
    try:
        GROQ_CLIENT = Groq(api_key=api_key)
    except Exception:
        GROQ_CLIENT = None

# ---------- App ----------
app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Storage paths ----------
PROFILE_PATH = Path(__file__).parent.parent / "data" / "profile.json"
PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----- new: persisted weekly plan -----
PLAN_PATH = Path(__file__).parent.parent / "data" / "plan.json"
PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- Schemas ----------
WorkoutStyle = Literal["cardio", "hiit", "yoga", "strength"]
Difficulty   = Literal["easy", "medium", "hard"]

Gender       = Literal["female", "male", "nonbinary", "unspecified"]
FitnessLevel = Literal["beginner", "intermediate", "expert"]
Goal         = Literal["lose_weight", "get_fit", "get_stronger", "more_flexible", "gain_muscle"]

Age        = Annotated[int,   Field(ge=10,  le=90)]
HeightCm   = Annotated[int,   Field(ge=120, le=230)]
WeightKg   = Annotated[float, Field(ge=30,  le=250)]
Duration   = Annotated[int,   Field(ge=5,   le=180)]
SessionMin = Annotated[int,   Field(ge=10,  le=120)]

class SchedulePref(BaseModel):
    days: List[Literal["mon","tue","wed","thu","fri","sat","sun"]] = Field(
        default_factory=lambda: ["mon","wed","fri"]
    )
    time_of_day: Literal["morning", "afternoon", "evening"] = "evening"
    session_minutes: SessionMin = 30

class Profile(BaseModel):
    # basics
    name: str = "Athlete"
    age: Age = 30
    height_cm: HeightCm = 175
    weight_kg: WeightKg = 75
    gender: Gender = "unspecified"
    # training prefs
    preferred_style: WorkoutStyle = "strength"
    fitness_level: FitnessLevel = "beginner"
    goal: Goal = "get_fit"
    schedule: SchedulePref = SchedulePref()

class QuickPredictIn(BaseModel):
    workout_type: WorkoutStyle = "strength"
    difficulty: Difficulty = "medium"
    duration: Duration = 30
    age: Optional[Age] = None
    weight_kg: Optional[WeightKg] = None
    height_cm: Optional[HeightCm] = None
    heart_rate: Optional[int] = 145

class QuickPredictOut(BaseModel):
    total_calories: float

# ----- Generation (sessions + meals) -----
Intensity = Literal["easy", "moderate", "hard"]

class Exercise(BaseModel):
    name: str
    sets: int
    reps: int

class PlanSession(BaseModel):
    day: Literal["mon","tue","wed","thu","fri","sat","sun"]
    type: str               # e.g., "Cardio", "Strength"
    intensity: Intensity    # easy | moderate | hard
    duration: int           # minutes
    exercises: List[Exercise]
    # NEW: calories predicted by our model for this session
    target_calories: Optional[float] = None

class MealDay(BaseModel):
    day: Literal["mon","tue","wed","thu","fri","sat","sun"]
    meals: List[str]

class GenerateIn(BaseModel):
    allergy: Optional[str] = None

class GenerateOut(BaseModel):
    sessions: List[PlanSession]
    meals: List[MealDay]

# ---------- Helpers ----------
def _read_profile() -> Profile:
    if PROFILE_PATH.exists():
        return Profile(**json.loads(PROFILE_PATH.read_text()))
    return Profile()

def _write_profile(p: Profile) -> None:
    PROFILE_PATH.write_text(json.dumps(p.model_dump(), indent=2))

def _normalize_workout_type(name: str, default_style: str) -> WorkoutStyle:
    s = (name or "").strip().lower()
    if "cardio" in s: return "cardio"
    if "hiit" in s: return "hiit"
    if "yoga" in s: return "yoga"
    if "strength" in s: return "strength"
    # fallback to profile style
    ds = (default_style or "strength").lower()
    return "cardio" if ds == "cardio" else "hiit" if ds == "hiit" else "yoga" if ds == "yoga" else "strength"

def _intensity_to_difficulty(intensity: str) -> Difficulty:
    s = (intensity or "").lower()
    if s == "easy": return "easy"
    if s == "moderate": return "medium"
    return "hard"

def _hr_for_intensity(intensity: str) -> int:
    # simple defaults; change if you have better HR estimation
    s = (intensity or "").lower()
    return 120 if s == "easy" else 145 if s == "moderate" else 165

def _attach_calories(profile: Profile, sessions: List[PlanSession]) -> List[PlanSession]:
    """Compute target_calories for each session using predict_calories_burned_simple."""
    out: List[PlanSession] = []
    for ses in sessions:
        wt = _normalize_workout_type(ses.type, profile.preferred_style)
        diff = _intensity_to_difficulty(ses.intensity)
        hr = _hr_for_intensity(ses.intensity)

        total = predict_calories_burned_simple(
            workout_type=wt,
            difficulty=diff,
            duration_minutes=int(ses.duration),
            age=int(profile.age),
            weight=float(profile.weight_kg),
            height=int(profile.height_cm),
            heart_rate=hr,
        )
        # copy with calories
        ses_with_cals = ses.model_copy(update={"target_calories": float(round(total, 1))})
        out.append(ses_with_cals)
    return out

def fallback_generate(profile: Profile, allergy: Optional[str]) -> GenerateOut:
    days = profile.schedule.days or ["mon","wed","fri"]
    mins = int(profile.schedule.session_minutes)
    style = profile.preferred_style
    fit   = profile.fitness_level
    age   = int(profile.age)
    goal  = getattr(profile, "goal", "get_fit")  # safe default

    lvl2int = {"beginner":"easy","intermediate":"moderate","expert":"hard"}
    base_int: Intensity = lvl2int.get(fit, "easy")  # type: ignore

    if fit == "expert":
        sets = 4; reps = 12
    elif fit == "intermediate":
        sets = 3; reps = 10
    else:
        sets = 3; reps = 8
    if age >= 55:
        reps = max(6, reps - 2)

    def generic_exercises(style_name: str) -> List[Exercise]:
        if style_name == "cardio":
            return [
                Exercise(name="Squats", sets=sets, reps=reps),
                Exercise(name="Push-ups", sets=sets, reps=reps),
            ]
        if style_name == "strength":
            return [
                Exercise(name="Squats", sets=sets, reps=reps),
                Exercise(name="Push-ups", sets=sets, reps=reps),
            ]
        if style_name == "hiit":
            return [
                Exercise(name="Burpees", sets=sets, reps=reps),
                Exercise(name="Mountain Climbers", sets=sets, reps=reps),
            ]
        # yoga
        return [
            Exercise(name="Sun Salutation", sets=max(2, sets-1), reps=8),
            Exercise(name="Chair Pose Hold (sec)", sets=max(2, sets-1), reps=30),
        ]

    sessions: List[PlanSession] = []
    for idx, d in enumerate(days):
        intensity: Intensity = base_int
        if fit != "beginner" and idx >= 1:
            intensity = "moderate" if base_int == "easy" else "hard"  # type: ignore

        if style == "cardio" and idx == 0:
            exs = [
                Exercise(name="Squats", sets=sets, reps=reps),
                Exercise(name="Push-ups", sets=sets, reps=reps),
            ]
        else:
            exs = generic_exercises(style)

        sessions.append(
            PlanSession(
                day=d,
                type=style.capitalize(),
                intensity=intensity,
                duration=mins,
                exercises=exs,
            )
        )

    # ⬇️ attach predicted calories to sessions
    sessions = _attach_calories(profile, sessions)

    raw_meals = {
        "mon": ["Greek yogurt + berries", "Grilled chicken salad", "Salmon, quinoa, broccoli"],
        "tue": ["Overnight oats", "Turkey wrap + veggies", "Stir-fry tofu + rice"],
        "wed": ["Egg scramble + spinach", "Lentil soup + side salad", "Beef + sweet potato + greens"],
        "thu": ["Protein smoothie", "Chickpea bowl", "Chicken fajita bowl"],
        "fri": ["Avocado toast + eggs", "Sushi or tuna bowl", "Pasta + chicken + veg"],
        "sat": ["Pancakes + fruit", "Burger (lean) + salad", "Shrimp + couscous + veg"],
        "sun": ["Omelet + veg", "Quinoa salad + feta", "Roast chicken + potatoes + carrots"],
    }
    meals: List[MealDay] = []
    for d, items in raw_meals.items():
        filtered = items
        if allergy:
            a = allergy.lower()
            filtered = [m for m in items if a not in m.lower()]
        meals.append(MealDay(day=d, meals=filtered))
    return GenerateOut(sessions=sessions, meals=meals)


DASHBOARD_PATH = Path(__file__).parent.parent / "data" / "dashboard.json"
DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

WORKOUT_JSON_RE = re.compile(r"===\s*LLM_PROMPT:WORKOUT\s*===\s*(\{.*?\})\s*===\s*END\s*===", re.S)

def _run_script_and_grab_workout_json(cmd: list[str], env_overrides: dict[str, str]) -> dict:
    """Runs a script, mirrors stdout to server logs, returns parsed WORKOUT JSON if found."""
    env = os.environ.copy()
    env.update(env_overrides or {})

    # --- ensure unicode-safe stdout from child Python on Windows ---
    env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING", "utf-8")
    # (optional) quieter OpenCV + MediaPipe logs if you want:
    # env.setdefault("OPENCV_LOG_LEVEL", "SILENT")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_DIR),
        env=env,
    )
    captured = []
    try:
        for line in proc.stdout:  # type: ignore
            print(line.rstrip())
            captured.append(line)
    finally:
        proc.wait()

    blob = "".join(captured)
    m = WORKOUT_JSON_RE.search(blob)
    if not m:
        return {"type": "workout_summary", "error": "no_json_block_found"}
    try:
        return json.loads(m.group(1))
    except Exception as e:
        return {"type": "workout_summary", "error": f"json_parse_failed: {e}"}

class RunFirstSessionIn(BaseModel):
    # From your card UI:
    squat_sets: int
    squat_reps: int
    push_sets: int
    push_reps: int

# ---------- Routes ----------
@app.get("/health")
def health():
  return {"status": "ok", "app": settings.APP_NAME}

@app.get("/api/profile", response_model=Profile)
def get_profile():
  return _read_profile()

@app.post("/api/profile", response_model=Profile)
def save_profile(p: Profile):
  _write_profile(p)
  return p

@app.post("/api/predict", response_model=QuickPredictOut)
def quick_predict(inp: QuickPredictIn):
  prof = _read_profile()
  age = inp.age or prof.age
  weight = inp.weight_kg or prof.weight_kg
  height = inp.height_cm or prof.height_cm
  hr = inp.heart_rate or 145

  total = predict_calories_burned_simple(
      workout_type=inp.workout_type,
      difficulty=inp.difficulty,
      duration_minutes=inp.duration,
      age=age,
      weight=weight,
      height=height,
      heart_rate=hr,
  )
  return QuickPredictOut(total_calories=float(total))

@app.post("/api/generate", response_model=GenerateOut)
def generate_plan(inp: GenerateIn):
  prof = _read_profile()

  if GROQ_CLIENT is None:
      return fallback_generate(prof, inp.allergy)

  # ----------------- TEST OVERRIDE: force 2x2 Squats + Push-ups in every session -----------------
  rules = (
        "Generate a weekly training plan and a 7-day meal prep list. "
        "Use the user's profile (age, gender, fitness_level, preferred_style, schedule.days, schedule.session_minutes). "
        "HARD RULES: "
        "1) If preferred_style is 'cardio', EVERY session MUST include both Squats and Push-ups among exercises, "
        "   AND the FIRST session must include ONLY these two exercises: Squats and Push-ups (no other exercise names). "
        "2) Output STRICT JSON ONLY matching the schema. No commentary. "
        "3) The number of sessions equals the number of schedule.days. "
        "4) For each session JSON: {day, type, intensity, duration, exercises:[{name,sets,reps}]}. "
        "   - day must be one of mon,tue,wed,thu,fri,sat,sun. "
        "   - intensity is one of easy, moderate, hard. "
        "   - duration is an integer in minutes. "
        "5) Meal plan: array of 7 days, each {day, meals:[...]}. Avoid any allergy ingredients provided."
    )
  
  # -----------------------------------------------------------------------------------------------

  profile_blob = prof.model_dump()

  system = "You are a precise planner that returns strict JSON matching the requested schema."
  user = {
      "profile": profile_blob,
      "allergy": inp.allergy or "",
      "requirements": rules,
      "json_schema": {
          "type": "object",
          "properties": {
              "sessions": {
                  "type": "array",
                  "items": {
                      "type": "object",
                      "properties": {
                          "day": {"type":"string", "enum":["mon","tue","wed","thu","fri","sat","sun"]},
                          "type": {"type":"string"},
                          "intensity": {"type":"string", "enum":["easy","moderate","hard"]},
                          "duration": {"type":"integer"},
                          "exercises": {
                              "type":"array",
                              "items": {
                                  "type":"object",
                                  "properties":{
                                      "name":{"type":"string"},
                                      "sets":{"type":"integer"},
                                      "reps":{"type":"integer"}
                                  },
                                  "required":["name","sets","reps"]
                              }
                          }
                      },
                      "required":["day","type","intensity","duration","exercises"]
                  }
              },
              "meals": {
                  "type": "array",
                  "items": {
                      "type":"object",
                      "properties":{
                          "day": {"type":"string", "enum":["mon","tue","wed","thu","fri","sat","sun"]},
                          "meals": {"type":"array","items":{"type":"string"}}
                      },
                      "required":["day","meals"]
                  }
              }
          },
          "required":["sessions","meals"]
      }
  }

  try:
      chat = GROQ_CLIENT.chat.completions.create(
          model=model,
          temperature=0.2,
          response_format={"type": "json_object"},
          messages=[
              {"role": "system", "content": system},
              {"role": "user", "content": json.dumps(user)}
          ],
      )
      content = chat.choices[0].message.content
      data = json.loads(content)

      sessions = [PlanSession(**s) for s in data.get("sessions", [])]
      meals = [MealDay(**m) for m in data.get("meals", [])]

      # ⬇️ attach predicted calories using your model
      sessions = _attach_calories(prof, sessions)
      # persist last generated plan so frontend Showcase can read it later
      plan_payload = {"sessions": [s.model_dump() for s in sessions],
                "meals": [m.model_dump() for m in meals]}
      PLAN_PATH.write_text(json.dumps(plan_payload, indent=2))

      return GenerateOut(sessions=sessions, meals=meals)
  except Exception:
      return fallback_generate(prof, inp.allergy)



@app.post("/api/run-first-session")
def run_first_session(inp: RunFirstSessionIn):
    """
    1) Launch squat.py with SET_SIZE/NUM_SETS from the first card.
    2) On exit, launch push.py with its sets/reps.
    3) Merge the two workout summaries into dashboard.json.
    """
    # ---- 1) SQUAT ----
    squat_env = {
        "SET_SIZE": str(inp.squat_reps),
        "NUM_SETS": str(inp.squat_sets),
        # keep rest short-ish so UX is snappy; or pass nothing to use each file’s default
        "REST_SEC": os.environ.get("REST_SEC", "15"),
    }
    squat_cmd = [PYTHON_EXE, str(ROOT_DIR / "squat.py")]   # <<< use venv python + absolute path
    squat_summary = _run_script_and_grab_workout_json(squat_cmd, squat_env)

    delay_sec = float(os.environ.get("RUN_CHAIN_DELAY_SEC", "2.5"))  # tweak via env if needed
    time.sleep(delay_sec)

    # ---- 2) PUSH ----
    push_env = {
        "SET_SIZE": str(inp.push_reps),
        "NUM_SETS": str(inp.push_sets),
        "REST_SEC": os.environ.get("REST_SEC", "15"),
    }
    push_cmd = [PYTHON_EXE, str(ROOT_DIR / "push.py")]     # <<< use venv python + absolute path
    push_summary = _run_script_and_grab_workout_json(push_cmd, push_env)

    # ---- 3) Combine + save dashboard.json ----
    dashboard = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session": {
            "first_session": {
                "squat": {"sets": inp.squat_sets, "reps": inp.squat_reps},
                "push":  {"sets": inp.push_sets,  "reps": inp.push_reps},
            }
        },
        "summaries": {
            "squat": squat_summary,
            "push":  push_summary,
        },
        # helpful derived label if both have a label/score
        "quick": {
            "squat_label": squat_summary.get("verdict") or squat_summary.get("quality",{}).get("final_label"),
            "push_label":  push_summary.get("verdict")  or push_summary.get("quality",{}).get("final_label"),
        }
    }
    DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2))
    return dashboard


@app.get("/api/dashboard")
def get_dashboard():
    if DASHBOARD_PATH.exists():
        return json.loads(DASHBOARD_PATH.read_text())
    return {"status": "empty"}


@app.get("/api/plan")
def get_plan():
    if PLAN_PATH.exists():
        return json.loads(PLAN_PATH.read_text())
    # empty shape for frontend fallback
    return {"sessions": [], "meals": []}



# Run:
# uvicorn app.main:app --reload --app-dir backend
