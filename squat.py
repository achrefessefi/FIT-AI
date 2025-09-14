# squat.py (compact HUD + hotkeys + cleaner overlays)
import cv2, math, time, numpy as np, os, csv, sys, json, threading, random
from datetime import datetime
from dotenv import load_dotenv

# Load .env from current working directory
load_dotenv()

print("[BOOT] Starting Squat Coach…")

# ---------------- Camera opener (Windows-friendly) ----------------
def try_open():
    for flag in [cv2.CAP_DSHOW, 0]:
        for idx in [0,1,2]:
            cap = cv2.VideoCapture(idx, flag) if flag else cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap, idx, ("CAP_DSHOW" if flag==cv2.CAP_DSHOW else "DEFAULT")
            cap.release()
    return None, None, None

# ---------------- Threaded latest-frame grabber ----------------
class FrameGrabber:
    """Reads camera in a thread and keeps only the newest frame to avoid lag."""
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.last_ts = None
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return self

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            ts = time.time()
            if not ok:
                time.sleep(0.001)
                continue
            with self.lock:
                self.frame = f
                self.last_ts = ts

    def read(self):
        with self.lock:
            return (self.frame.copy() if self.frame is not None else None), self.last_ts

    def stop(self):
        self.stopped = True

# ---------------- Math helpers ----------------
def angle_deg(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    v1=np.array([ax-bx, ay-by], np.float32)
    v2=np.array([cx-bx, cy-by], np.float32)
    n1=np.linalg.norm(v1)+1e-6; n2=np.linalg.norm(v2)+1e-6
    cos=np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
    return float(np.degrees(np.arccos(cos)))

def trunk_lean_deg(hip, shoulder):
    vx,vy=shoulder[0]-hip[0], shoulder[1]-hip[1]
    mag=max(1e-6, math.hypot(vx,vy))
    cos=np.clip((vy*-1)/mag, -1, 1)  # angle vs vertical
    return float(np.degrees(np.arccos(cos)))

def l2(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

class EMA:
    def __init__(self,a): self.a=a; self.y=None
    def push(self,x): self.y=x if self.y is None else self.y+(x-self.y)*self.a; return self.y

# ---------------- .env parsing helpers ----------------
def _env_bool(key, default=False):
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def _env_int(key, default):
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _env_float(key, default):
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def _env_str(key, default):
    v = os.getenv(key)
    return v if (v is not None and v.strip() != "") else default

# ---------------- Config ----------------
CFG = {
    # Posture / thresholds (snappy & permissive)
    "knee_top_deg": 170,
    "knee_bottom_deg": 90,
    "trunk_lean_max_deg": 40,

    # Enter/exit phases
    "TOP_ENTER_DEG": 165,
    "BOTTOM_ENTER_DEG": 135,
    "BOTTOM_EXIT_DELTA": 1,
    "ROM_DROP_MIN": 3.0,

    # Time guards
    "min_up_s": 0.03,
    "min_down_s": 0.03,
    "min_cycle_s": 0.20,
    "require_bottom_hold_s": 0.00,

    # Smoothing
    "ema_alpha": 0.35,

    # Visibility
    "vis_conf_thresh": 0.40,
    "min_visible_pct": 0.50,
    "require_key_joints": True,

    # Live intent & cadence
    "intent_gain": 3.9,
    "intent_floor_ratio": 0.6,
    "intent_ceiling": 100.0,
    "cad_gain": 3.2,
    "cad_target_mult": 0.90,
    "cad_target_min": 0.90,
    "cad_target_max": 2.00,

    # Rep motivation
    "mot_gain_pos": 85.0,
    "mot_gain_neg": 15.0,
    "mot_idle_ref": 10.0,
    "mot_depth_ref_deg": 24.0,
    "cov_ref": 0.35,

    # Labels
    "label_good": _env_int("LABEL_GOOD", 55),
    "label_neutral": _env_int("LABEL_NEUTRAL", 45),

    # Sets & workout
    "SET_SIZE": _env_int("SET_SIZE", 3),
    "NUM_SETS": _env_int("NUM_SETS", 3),
    "REST_SEC": _env_int("REST_SEC", 15),

    # Calories estimate
    "CAL_PER_SQUAT": _env_float("CAL_PER_SQUAT", 0.5),

    # Groq LLM
    "GROQ_ENABLE": _env_bool("GROQ_ENABLE", True),
    "GROQ_MODEL": _env_str("GROQ_MODEL", "llama-3.1-70b-versatile"),

    # UI / Cam
    "DRAW_EVERY_N": 1,

    # Camera target
    "TARGET_FPS": _env_int("TARGET_FPS", 60),
    "CAM_W": _env_int("CAM_W", 960),
    "CAM_H": _env_int("CAM_H", 540),
    "POSE_INPUT_W": _env_int("POSE_INPUT_W", 640),

    # Velocity blend (hip + knee)
    "USE_KNEE_SPEED_BLEND": True,
    "KNEE_DPS_TO_LEGLEN_SCALE": 1.0/180.0,
    "BLEND_MAX": True,

    # Logging
    "LOG_EVERY_N_FRAMES": 3,
    "CSV_PATH": _env_str("CSV_PATH", "performance_log.csv"),

    # ====== TTS ======
    "TTS_ENABLE": _env_bool("TTS_ENABLE", True),
    "TTS_VOICE": _env_str("TTS_VOICE", ""),
    "TTS_RATE": _env_int("TTS_RATE", 185),
    "TTS_VOLUME": _env_float("TTS_VOLUME", 1.0),

    # ====== HUD ======
    # Modes: "mini", "full", "off"
    "HUD_MODE": _env_str("HUD_MODE", "mini"),
    "HUD_ALPHA": _env_float("HUD_ALPHA", 0.35),   # panel transparency
    "HUD_SIDEBAR_W": _env_int("HUD_SIDEBAR_W", 260)
}

# ---------------- CSV logging ----------------
CSV_HEADERS = [
    "ts","set_idx","rep_idx","state",
    "live_up_norm","live_cadence","last_down_vel_deg_s","last_up_vel_deg_s",
    "cycle_s","idle_top_s","depth_min_knee_deg",
    "mot_live","mot_live_label","mot_rep","mot_rep_label"
]

def ensure_csv(path):
    new = not os.path.exists(path)
    if new:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f); writer.writerow(CSV_HEADERS)

def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ---------------- LLM prompt helpers ----------------
def _verdict_from_mot(avg_mot):
    if   avg_mot >= 70: return "Good"
    elif avg_mot >= 60: return "Medium"
    elif avg_mot >= 50: return "Could do better"
    else:               return "Bad"

def _qualitative_notes(avg_up_norm, avg_down_dps, rpm):
    notes = []
    if avg_up_norm >= 0.95: notes.append("ascent_fast")
    elif avg_up_norm >= 0.70: notes.append("ascent_moderate")
    else: notes.append("ascent_slow")
    if avg_down_dps >= 90: notes.append("descent_very_fast")
    elif avg_down_dps >= 60: notes.append("descent_fast")
    elif avg_down_dps >= 30: notes.append("descent_controlled")
    else: notes.append("descent_slow")
    if rpm >= 18: notes.append("cadence_high")
    elif rpm >= 10: notes.append("cadence_moderate")
    else: notes.append("cadence_low")
    return notes

def _coaching_cues(notes):
    cues = []
    if "descent_very_fast" in notes:
        cues.append("Control the descent—avoid dropping and bouncing.")
    elif "descent_fast" in notes:
        cues.append("Slightly slow the eccentric for consistency.")
    if "ascent_slow" in notes:
        cues.append("Drive up faster—push the floor, brace hard.")
    if "cadence_low" in notes:
        cues.append("Reduce rest at the top to keep rhythm.")
    if not cues:
        cues.append("Great pace—keep the same intent on each ascent.")
    return cues

def _now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _system_prompt():
    return (
        "You are an energetic, professional strength coach.\n"
        "Given a squat metrics JSON, write 2–4 SHORT sentences that:\n"
        "• Celebrate the athlete's wins using their numbers.\n"
        "• Give 1–2 PRECISE cues tailored to the provided 'focus_theme'.\n"
        "• End with a motivating call-to-action.\n"
        "STYLE: concise, varied, positive; prioritize safety if stance/knee flags appear.\n"
    )

def _user_prompt_from_payload(payload):
    header = (
        "Athlete just finished a set/workout. Offer clear, upbeat feedback and motivation.\n"
        "Keep it tight and powerful.\n\n"
    )
    return header + json.dumps(payload, ensure_ascii=False)

# ---- TTS Worker (non-blocking, with draining) ----
# ---- TTS Worker (non-blocking, with draining) ----
class TTSWorker:
    def __init__(self, cfg):
        self.enabled = cfg.get("TTS_ENABLE", True)
        self.queue = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.thread = None
        self.engine = None
        if self.enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()

                # --- Voice selection: prefer English or explicit TTS_VOICE
                chosen = None
                want = (cfg.get("TTS_VOICE") or "").strip().lower()
                voices = self.engine.getProperty("voices")

                def is_en(v):
                    try:
                        langs = getattr(v, "languages", []) or []
                        langs = [str(x).lower() for x in langs]
                    except Exception:
                        langs = []
                    name = (v.name or "").lower()
                    vid  = (getattr(v, "id", "") or "").lower()
                    return ("en" in "".join(langs)) or ("english" in name) or ("en_" in vid) or ("english" in vid)

                if want:
                    for v in voices:
                        if want in (v.name or "").lower() or want in (getattr(v, "id", "") or "").lower():
                            chosen = v
                            break
                if chosen is None:
                    for v in voices:
                        if is_en(v):
                            chosen = v
                            break
                if chosen:
                    self.engine.setProperty("voice", chosen.id)
                    print(f"[TTS] Using voice: {chosen.name}")
                else:
                    print("[TTS] No English voice found; using default system voice.")

                self.engine.setProperty("rate", int(cfg.get("TTS_RATE", 185)))
                self.engine.setProperty("volume", float(cfg.get("TTS_VOLUME", 1.0)))

                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
                print("[TTS] Initialized.")
            except Exception as e:
                print("[TTS] init failed:", repr(e))
                self.enabled = False

    def _run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                text = self.queue.pop(0)
            try:
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                print("[TTS] speak error:", repr(e))

    def say(self, text):
        if not self.enabled or not text:
            return
        with self.cv:
            self.queue.append(text)
            self.cv.notify()

    def wait_empty(self, timeout_sec=20.0):
        """Block until engine and queue are empty, or timeout."""
        if not self.enabled or self.engine is None:
            return
        start = time.time()
        try:
            while time.time() - start < timeout_sec:
                with self.cv:
                    q_empty = (len(self.queue) == 0)
                busy = getattr(self.engine, "isBusy", lambda: False)()
                if q_empty and not busy:
                    return
                time.sleep(0.05)
        except Exception:
            pass


# ---------------- Groq send (returns text for TTS) ----------------
def _send_to_groq(kind, payload, cfg):
    if not cfg.get("GROQ_ENABLE", False):
        print("[GROQ] Disabled by config; not sending.")
        return None
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[GROQ] GROQ_API_KEY not set; skipping LLM send.")
            return None
        client = Groq(api_key=api_key)
        model = cfg.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt_from_payload(payload)},
            ],
            temperature=0.8,
            max_tokens=400,
        )
        text = rsp.choices[0].message.content.strip()
        tag = "SET" if kind == "set" else "WORKOUT"
        print(f"=== LLM_FEEDBACK:{tag} ===")
        print(text)
        print("=== END ===")
        return text
    except Exception as e:
        print("[GROQ] Error sending to Groq:", repr(e))
        return None

# ---------------- Motivation explanations ----------------
def _diagnose_motivation(avg_up_norm, avg_down_dps, rpm, depth_deg=None, idle_top_avg=None, cov=None):
    notes = []
    if avg_up_norm < 0.70:
        notes.append("Drive up faster: ascent speed low (< 0.70).")
    elif avg_up_norm < 0.95:
        notes.append("Ascent speed moderate (0.70–0.95) — push for a snappier stand-up.")
    else:
        notes.append("Ascent speed is strong (≥ 0.95).")
    if avg_down_dps >= 110:
        notes.append("Control the drop: descent extremely fast (≥ 110°/s).")
    elif avg_down_dps >= 90:
        notes.append("Control the drop: descent very fast (90–110°/s).")
    elif avg_down_dps >= 30:
        notes.append("Descent speed controlled (30–90°/s).")
    else:
        notes.append("Descent very slow (< 30°/s) — you can lower a bit faster while staying controlled.")
    if rpm < 8:
        notes.append("Tighten rhythm: long pause between reps (cadence < 8 rpm).")
    elif rpm < 10:
        notes.append("Cadence a bit low (8–10 rpm) — reduce top pause.")
    elif rpm <= 18:
        notes.append("Cadence is solid (10–18 rpm).")
    else:
        notes.append("High cadence (> 18 rpm) — keep control at the bottom.")
    if depth_deg is not None:
        if depth_deg > 150: notes.append("Squat deeper: depth shallow (knee angle > 150°).")
        elif depth_deg > 145: notes.append("Depth a bit shallow (145–150°). Try a bit lower.")
        elif depth_deg > 135: notes.append("Depth okay (135–145°).")
        else: notes.append("Depth solid (≤ 135°).")
    if idle_top_avg is not None:
        if idle_top_avg > 2.5: notes.append("Top pause long on average (> 2.5s) — start next rep sooner.")
        elif idle_top_avg > 1.5: notes.append("Top pause moderate (1.5–2.5s).")
        else: notes.append("Top pause short (≤ 1.5s).")
    if cov is not None:
        if cov > 0.45: notes.append("Reps timing inconsistent (high variability) — aim for steady tempo.")
        elif cov > 0.30: notes.append("Timing varies a bit — try to keep rep tempo consistent.")
        else: notes.append("Good consistency between reps.")
    return notes

# ---------------- Visibility utilities ----------------
def percent_visible(lms, vis_conf_thresh=0.5):
    if not lms: return 0.0
    visible = sum(1 for lm in lms if getattr(lm, "visibility", 0.0) >= vis_conf_thresh)
    return visible / float(len(lms))

def joints_visible(lms, idxs, vis_conf_thresh=0.5):
    try:
        return all(lms[int(i)].visibility >= vis_conf_thresh for i in idxs)
    except Exception:
        return False

# ---------------- HUD helpers (new compact UI) ----------------
def _panel(frame, x, y, w, h, alpha):
    """Semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (25,25,25), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (70,70,70), 1)

def draw_bar(frame, x, y, w, h, frac, color_fg, color_bg=(40,40,40)):
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    cv2.rectangle(frame, (x,y), (x+w, y+h), color_bg, -1)
    cv2.rectangle(frame, (x,y), (x+int(w*frac), y+h), color_fg, -1)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (90,90,90), 1)

def color_from_label(lbl):
    return (0,255,0) if lbl=="Motivated" else ((0,215,255) if lbl=="Neutral" else (0,0,255))

def draw_hud(frame, info, set_idx, total_sets, rest_left=None, workout_done=False,
             good_in_set=None, total_in_set=None, mode="mini", alpha=0.35, side_w=260):
    h,w=frame.shape[:2]
    pad = 12

    if mode == "off":
        return

    if mode == "mini":
        # Right sidebar only (small)
        x = w - side_w - pad
        y = pad
        _panel(frame, x, y, side_w, 160, alpha)

        title = "DONE" if workout_done else f"Set {set_idx}/{total_sets}"
        cv2.putText(frame, title, (x+10, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        # Reps + good ratio
        rep_txt = f"Reps: {info['rep']}"
        cv2.putText(frame, rep_txt, (x+10, y+48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if good_in_set is not None and total_in_set is not None and total_in_set > 0:
            rate = 100.0 * good_in_set / float(total_in_set)
            cv2.putText(frame, f"Good: {good_in_set}/{total_in_set} ({rate:.0f}%)",
                        (x+10, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2)

        # Live motivation bar
        live_lbl = info.get("live_label","Neutral")
        live_val = float(info.get("live_mot", 50.0))
        live_col = color_from_label(live_lbl)
        cv2.putText(frame, f"Motivation: {live_lbl} ({int(live_val)})",
                    (x+10, y+94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, live_col, 2)
        draw_bar(frame, x+10, y+100, side_w-20, 12, live_val/100.0, live_col)

        # Rest or short msg
        cal_now = info.get("calories_now", 0.0)
        msg = ""
        if rest_left is not None and rest_left > 0:
            msg = f"REST {int(rest_left)}s"
        elif workout_done:
            msg = "Workout complete"
        else:
            msg = info["msg"]

        cv2.putText(frame, msg, (x+10, y+126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Calories: {cal_now:.1f}",
                    (x+10, y+148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)
        return

    # ---- FULL mode (tightened version of your original) ----
    box_h = 230
    _panel(frame, 10, 10, 680, box_h, alpha)

    title = "Workout DONE" if workout_done else f"Set {set_idx}/{total_sets}"
    cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"REPS: {info['rep']}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    if good_in_set is not None and total_in_set is not None and total_in_set > 0:
        rate = 100.0 * good_in_set / float(total_in_set)
        cv2.putText(frame, f"Good: {good_in_set}/{total_in_set} ({rate:.0f}%)",
                    (180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)

    dv = info.get("down_vel", 0.0)
    uv = info.get("up_vel", 0.0)
    cyc = info.get("cycle", 0.0)
    depth = info.get("depth", 0.0)
    live_mot = info.get("live_mot", 50.0)
    live_lbl = info.get("live_label", "Neutral")
    mot_label = info.get("mot_label","Neutral")
    mot_score = info.get("mot_score",50)
    cal_now = info.get("calories_now", 0.0)

    cv2.putText(frame, f"Down: {dv:5.1f}°/s  Up: {uv:5.1f}°/s  Cycle: {cyc:4.2f}s  Depth: {depth:5.1f}°",
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    live_color = color_from_label(live_lbl)
    cv2.putText(frame, f"Live Mot: {live_lbl} ({int(live_mot)})", (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, live_color, 2)
    draw_bar(frame, x=20, y=112, w=520-40, h=12, frac=live_mot/100.0, color_fg=live_color)

    mot_color = color_from_label(mot_label)
    cv2.putText(frame, f"Rep Mot: {mot_label} ({mot_score})", (20, 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mot_color, 2)
    draw_bar(frame, x=20, y=142, w=520-40, h=12, frac=mot_score/100.0, color_fg=mot_color)

    try:
        per_rep = float(CFG.get("CAL_PER_SQUAT", 0.5))
    except Exception:
        per_rep = 0.5
    cv2.putText(frame, f"Calories: {cal_now:.1f} kcal (≈{per_rep:.2f}/rep)",
                (20, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    hint = info.get("hint")
    if hint:
        cv2.putText(frame, f"Hint: {hint}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

    # Bottom status ribbon (thin)
    status = info["msg"]
    if rest_left is not None and rest_left > 0:
        status = f"REST… {int(rest_left)}s"
    if workout_done:
        status = "Workout complete! Check console for summary."

    # Small ribbon bottom-left
    rb_h = 28
    _panel(frame, 10, h - rb_h - 10, 420, rb_h, alpha)
    cv2.putText(frame, status, (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["col"], 2, cv2.LINE_AA)

# ---------------- Coach ----------------
class SquatCoach:
    def __init__(self,cfg):
        self.c=cfg
        self.state="TOP"
        self.rep=0

        self.t_state=time.time()
        self.t_rep=time.time()
        self.t_bottom=None

        self.eL=EMA(cfg["ema_alpha"]); self.eR=EMA(cfg["ema_alpha"]); self.eT=EMA(cfg["ema_alpha"])
        self.msg="Ready"; self.col=(0,255,0)

        self.up_vels=[]; self.down_vels=[]; self.rep_times=[]; self.rom_mins=[]
        self.idle_top_s=0.0; self.last_top_time=time.time()

        self.last_down_vel=0.0; self.last_up_vel=0.0
        self.last_cycle=0.0; self.last_depth=0.0

        self.prev_k=None; self.prev_t=None
        self.top_k_start=None; self.drop_from_top=0.0

        self.prev_hip_y=None; self.prev_time=None
        self.up_vel_norm_frame=0.0
        self.up_vel_norm_rep=None
        self.up_vel_baseline=None

        self.baseline_cycle=None
        self.cadence_live=50.0

        self.motivation_score=50
        self.motivation_label="Neutral"
        self.live_mot=50.0
        self.live_label="Neutral"

        self._des_start=None; self._asc_start=None
        self._knee_at_top=None; self._knee_at_bottom=None
        self._rom_min_curr=None

        self.set_idx = 1
        self.reps_in_set = 0
        self.in_rest = False
        self.rest_end_time = None
        self.workout_done = False
        self.exit_now = False

        self._rep_up_norm_history = []
        self._rep_down_history = []
        self._rep_cycle_history = []
        self._rep_mot_history = []

        self._idle_history = []
        self._depth_history = []

        self.good_reps_total = 0
        self.good_reps_in_set = 0

        self.last_llm_set_text = None
        self.last_llm_workout_text = None

        # --- Calories tracking
        self.cal_per_rep = float(cfg.get("CAL_PER_SQUAT", 0.5))
        self.calories_now = 0.0

        ensure_csv(self.c["CSV_PATH"])
        self.frame_idx = 0

    def _median(self, arr): return float(np.median(arr)) if arr else None
    def _cov(self, arr):
        if len(arr) < 3: return 0.0
        m=float(np.mean(arr)); s=float(np.std(arr))
        return 0.0 if m<=1e-6 else float(s/m)
    def _clamp(self,x,a,b): return a if x<a else (b if x>b else x)

    # ---- blended hip+knee normalized up-speed for live intent ----
    def _blend_up_velocity_norm(self, pts, now, knee_up_dps):
        hipL, hipR = pts["l_hip"], pts["r_hip"]
        ankL, ankR = pts["l_ankle"], pts["r_ankle"]
        hip_y = 0.5*(hipL[1]+hipR[1])
        leg = max(1e-3, 0.5*(l2(hipL, ankL)+l2(hipR, ankR)))
        if self.prev_hip_y is not None and self.prev_time is not None:
            dt = max(1e-3, now - self.prev_time)
            v_y = (hip_y - self.prev_hip_y) / dt   # px/s (down +)
            up_norm_hip = (-v_y) / leg             # leg-lengths/s (up +)
        else:
            up_norm_hip = 0.0
        self.prev_hip_y, self.prev_time = hip_y, now

        knee_norm = (knee_up_dps * self.c["KNEE_DPS_TO_LEGLEN_SCALE"]) if self.c["USE_KNEE_SPEED_BLEND"] else 0.0
        self.up_vel_norm_frame = max(up_norm_hip, knee_norm) if self.c["BLEND_MAX"] else 0.5*(up_norm_hip + knee_norm)

    def _update_live_intent(self, pts, now, kmin=None, knee_up_dps=0.0):
        self._blend_up_velocity_norm(pts, now, knee_up_dps)
        if self.in_rest:
            self.live_mot = 50.0
            self.live_label = "Neutral"
            return

        if self.state == "ASCENT":
            base = self.up_vel_baseline or max(0.20, abs(self.up_vel_norm_frame))
            base = max(base, self.c["intent_floor_ratio"]*base)
            ratio = self.up_vel_norm_frame / (base + 1e-6)
            intent = 50.0 + 50.0 * math.tanh(self.c["intent_gain"] * (ratio - 1.0))
            self.live_mot = self._clamp(intent, 0.0, self.c["intent_ceiling"])
            self.live_label = "Motivated" if self.live_mot >= self.c["label_good"] else ("Neutral" if self.live_mot >= self.c["label_neutral"] else "Demotivated")
        elif self.state == "TOP":
            self.live_mot = 50.0
            self.live_label = "Neutral"

    def _update_live_cadence(self, now):
        if self.in_rest:
            self.cadence_live = 50.0
            return
        since_rep = now - self.t_rep
        if self.baseline_cycle is None:
            target = self.c["cad_target_min"]
        else:
            target = self._clamp(self.c["cad_target_mult"] * self.baseline_cycle,
                                 self.c["cad_target_min"], self.c["cad_target_max"])
        ratio = (target / max(0.1, since_rep))
        live = 50.0 + 50.0 * math.tanh(self.c["cad_gain"] * (ratio - 1.0))
        self.cadence_live = self._clamp(live, 0.0, 100.0)

    # ---- per-rep motivation ----
    def _update_motivation(self, last_cycle, last_rom_min):
        if self.up_vel_norm_rep is not None:
            if self.up_vel_baseline is None:
                self.up_vel_baseline = 0.75 * self.up_vel_norm_rep
            else:
                adapt = 0.15
                self.up_vel_baseline = (1 - adapt)*self.up_vel_baseline + adapt*self.up_vel_norm_rep

        if last_cycle and last_cycle > 0:
            self.baseline_cycle = 0.85*self.baseline_cycle + 0.15*last_cycle if self.baseline_cycle else last_cycle

        if self.up_vel_baseline is None or self.up_vel_baseline < 1e-6:
            self.motivation_score = 55
            self.motivation_label = "Neutral"
            return

        r = self.up_vel_norm_rep / (self.up_vel_baseline + 1e-6)
        if r >= 1.0: score = 50 + self.c["mot_gain_pos"] * (r - 1.0)
        else:        score = 50 - self.c["mot_gain_neg"] * (1.0 - r)

        idle_pen = self._clamp(self.idle_top_s / self.c["mot_idle_ref"], 0.0, 1.5)
        cov_pen  = self._clamp(self._cov(self.rep_times[-5:]) / self.c["cov_ref"], 0.0, 1.5)
        base_rom = (self._median(self.rom_mins[:3]) if len(self.rom_mins) >= 3
                    else (self.rom_mins[-1] if self.rom_mins else last_rom_min))
        base_rom = base_rom if base_rom is not None else last_rom_min
        depth_pen = self._clamp(max(0.0, (last_rom_min - base_rom)) / self.c["mot_depth_ref_deg"], 0.0, 1.5)

        score -= 5 * idle_pen
        score -= 4 * cov_pen
        score -= 6 * depth_pen

        score = int(self._clamp(score, 0, 100))
        self.motivation_score = score
        self.motivation_label = ("Motivated" if score >= self.c["label_good"]
                                 else ("Neutral" if score >= self.c["label_neutral"] else "Demotivated"))

    # ---- set summary / rest handling ----
    def _print_set_summary(self, set_idx):
        lastN = self.c["SET_SIZE"]
        ups   = self._rep_up_norm_history[-lastN:] if len(self._rep_up_norm_history) >= lastN else self._rep_up_norm_history[:]
        downs = self._rep_down_history[-lastN:]     if len(self._rep_down_history)     >= lastN else self._rep_down_history[:]
        cycles= self._rep_cycle_history[-lastN:]    if len(self._rep_cycle_history)    >= lastN else self._rep_cycle_history[:]
        mots  = self._rep_mot_history[-lastN:]      if len(self._rep_mot_history)      >= lastN else self._rep_mot_history[:]

        set_avg_up_norm = float(np.mean(ups))   if ups else 0.0
        set_avg_down    = float(np.mean(downs)) if downs else 0.0
        set_avg_cycle   = float(np.mean(cycles))if cycles else 0.0
        set_rpm         = (60.0/set_avg_cycle) if set_avg_cycle>1e-6 else 0.0
        set_avg_mot     = float(np.mean(mots))  if mots else 0.0
        set_verdict     = _verdict_from_mot(set_avg_mot)

        total_in_set = self.reps_in_set
        good_in_set  = self.good_reps_in_set
        good_rate_set = (100.0*good_in_set/total_in_set) if total_in_set>0 else 0.0

        depths = self._depth_history[-lastN:] if len(self._depth_history) >= lastN else self._depth_history[:]
        idles  = self._idle_history[-lastN:]  if len(self._idle_history)  >= lastN else self._idle_history[:]
        set_depth_med = float(np.median(depths)) if depths else None
        set_idle_avg  = float(np.mean(idles)) if idles else None
        set_cov       = self._cov(self._rep_cycle_history[-max(3,lastN):]) if self._rep_cycle_history else None

        explanations = _diagnose_motivation(
            avg_up_norm=set_avg_up_norm, avg_down_dps=set_avg_down, rpm=set_rpm,
            depth_deg=set_depth_med, idle_top_avg=set_idle_avg, cov=set_cov
        )

        print(f"\n=== Set {set_idx} complete ===")
        print(f"Avg Up Speed (norm): {set_avg_up_norm:.3f} leglen/s")
        print(f"Avg Down Speed:      {set_avg_down:.1f} deg/s")
        print(f"Avg Cadence:         {set_rpm:.1f} reps/min")
        print(f"Good Reps:           {good_in_set}/{total_in_set} ({good_rate_set:.0f}%)")
        print(f"Set Calories (est.): {total_in_set * self.cal_per_rep:.1f} kcal")
        if set_idx < self.c['NUM_SETS']:
            print(f"Rest {self.c['REST_SEC']}s…")
        print("Why this result:")
        for line in explanations:
            print(" -", line)

        notes = _qualitative_notes(set_avg_up_norm, set_avg_down, set_rpm)
        cues  = _coaching_cues(notes)
        payload = {
            "type": "set_summary",
            "timestamp_utc": _now_iso(),
            "session": {"target_sets": self.c["NUM_SETS"], "target_reps_per_set": self.c["SET_SIZE"]},
            "set_index": set_idx,
            "metrics": {
                "avg_up_speed_norm": round(set_avg_up_norm, 4),
                "avg_down_speed_deg_s": round(set_avg_down, 1),
                "avg_cadence_rpm": round(set_rpm, 1),
                "avg_motivation": round(set_avg_mot, 1),
                "good_rep_rate_pct": round(good_rate_set, 1),
            },
            "verdict": set_verdict,
            "notes": notes,
            "coaching_cues": cues,
            "explanations": explanations,
            "counts": {
                "reps_in_set": self.reps_in_set,
                "good_reps_in_set": self.good_reps_in_set,
                "total_reps_completed": self.rep,
                "recent_rep_motivation": [int(x) for x in self._rep_mot_history[-self.c["SET_SIZE"] :]]
            }
        }
        print("=== LLM_PROMPT:SET ===")
        print(json.dumps(payload, ensure_ascii=False))
        print("=== END ===")
        self.last_llm_set_text = _send_to_groq("set", payload, self.c)

    def _print_workout_summary_and_exit_flag(self):
        avg_up_norm = float(np.mean(self._rep_up_norm_history)) if self._rep_up_norm_history else 0.0
        avg_down    = float(np.mean(self._rep_down_history))    if self._rep_down_history    else 0.0
        avg_cycle   = float(np.mean(self._rep_cycle_history))   if self._rep_cycle_history   else 0.0
        avg_rpm     = (60.0/avg_cycle) if avg_cycle>1e-6 else 0.0
        avg_mot     = float(np.mean(self._rep_mot_history))     if self._rep_mot_history     else 0.0
        verdict     = _verdict_from_mot(avg_mot)

        total_reps = self.rep
        good_total = self.good_reps_total
        good_rate_total = (100.0*good_total/total_reps) if total_reps>0 else 0.0

        depth_med = float(np.median(self._depth_history)) if self._depth_history else None
        idle_avg  = float(np.mean(self._idle_history)) if self._idle_history else None
        cov_all   = self._cov(self._rep_cycle_history) if self._rep_cycle_history else None

        workout_explanations = _diagnose_motivation(
            avg_up_norm=avg_up_norm, avg_down_dps=avg_down, rpm=avg_rpm,
            depth_deg=depth_med, idle_top_avg=idle_avg, cov=cov_all
        )

        print("\n================ WORKOUT SUMMARY ================")
        print(f"Sets x Reps: {self.c['NUM_SETS']} x {self.c['SET_SIZE']}")
        print(f"Overall Avg Up Speed (norm): {avg_up_norm:.3f} leglen/s")
        print(f"Overall Avg Down Speed:      {avg_down:.1f} deg/s")
        print(f"Overall Avg Cadence:         {avg_rpm:.1f} reps/min")
        print(f"Overall Avg Motivation:      {avg_mot:.1f} → {verdict}")
        print(f"Good Reps (Total):           {good_total}/{total_reps} ({good_rate_total:.0f}%)")
        print(f"Total Calories (est.):       {self.calories_now:.1f} kcal")
        print("Why this result:")
        for line in workout_explanations:
            print(" -", line)
        print("=================================================\n")

        notes = _qualitative_notes(avg_up_norm, avg_down, avg_rpm)
        cues  = _coaching_cues(notes)
        payload = {
            "type": "workout_summary",
            "timestamp_utc": _now_iso(),
            "session": {"target_sets": self.c["NUM_SETS"], "target_reps_per_set": self.c["SET_SIZE"]},
            "totals": {"sets_completed": self.set_idx, "reps_completed": total_reps, "good_reps_total": self.good_reps_total},
            "metrics": {
                "avg_up_speed_norm": round(avg_up_norm, 4),
                "avg_down_speed_deg_s": round(avg_down, 1),
                "avg_cadence_rpm": round(avg_rpm, 1),
                "avg_motivation": round(avg_mot, 1),
                "good_rep_rate_pct": round((100.0*self.good_reps_total/total_reps) if total_reps>0 else 0.0, 1),
            },
            "verdict": _verdict_from_mot(avg_mot),
            "notes": notes,
            "coaching_cues": cues,
            "explanations": workout_explanations
        }
        print("=== LLM_PROMPT:WORKOUT ===")
        print(json.dumps(payload, ensure_ascii=False))
        print("=== END ===")
        self.last_llm_workout_text = _send_to_groq("workout", payload, self.c)

        self.workout_done = True
        self.in_rest = False
        self.rest_end_time = None
        self.exit_now = True

    def _maybe_finish_set_and_rest(self):
        if self.reps_in_set >= self.c["SET_SIZE"] and not self.in_rest:
            self._print_set_summary(self.set_idx)
            if self.set_idx >= self.c["NUM_SETS"]:
                self._print_workout_summary_and_exit_flag()
            else:
                self.in_rest = True
                self.rest_end_time = time.time() + self.c["REST_SEC"]

    def _maybe_end_rest(self):
        if self.exit_now:
            return True
        if self.in_rest and time.time() >= self.rest_end_time:
            self.in_rest = False
            self.rest_end_time = None
            self.set_idx += 1
            self.reps_in_set = 0
            self.good_reps_in_set = 0
        return False

    # ---- main kinematics + state machine ----
    def update(self,pts):
        now=time.time(); c=self.c

        if self.exit_now:
            return {"__exit__": True, "rep": self.rep, "msg": "Workout complete", "col": (0,255,0)}

        # angles
        kL=angle_deg(pts["l_hip"],pts["l_knee"],pts["l_ankle"])
        kR=angle_deg(pts["r_hip"],pts["r_knee"],pts["r_ankle"])
        kL=self.eL.push(kL); kR=self.eR.push(kR)
        tL=trunk_lean_deg(pts["l_hip"],pts["l_shoulder"])
        tR=trunk_lean_deg(pts["r_hip"],pts["r_shoulder"])
        trunk=self.eT.push(min(tL,tR))
        kmin=min(kL,kR)

        # knee angular velocity (deg/s)
        vel_k = 0.0
        if self.prev_k is not None and self.prev_t is not None:
            dt = max(1e-3, now - self.prev_t)
            vel_k = (kmin - self.prev_k) / dt
        self.prev_k, self.prev_t = kmin, now

        # live signals
        knee_up_dps = max(0.0, vel_k)  # positive when angle increases = ascent
        if not self.in_rest:
            self._update_live_intent(pts, now, kmin, knee_up_dps)
            self._update_live_cadence(now)
        else:
            self.live_mot = 50.0
            self.live_label = "Neutral"
            self.cadence_live = 50.0

        # min depth within rep
        if self._rom_min_curr is None: self._rom_min_curr = kmin
        else: self._rom_min_curr = min(self._rom_min_curr, kmin)

        def goto(s): self.state=s; self.t_state=now
        msg,col="Ready" if self.rep==0 else "Good",(0,255,0)

        if not self.in_rest:
            if self.state=="TOP":
                if kmin < c["TOP_ENTER_DEG"]:
                    self.idle_top_s = now - self.last_top_time
                    self._des_start = now
                    self._knee_at_top = kmin
                    self.top_k_start = kmin
                    self.drop_from_top = 0.0
                    self._rom_min_curr = kmin
                    goto("DESCENT"); msg="Descent"

            elif self.state=="DESCENT":
                if self.top_k_start is not None:
                    self.drop_from_top = max(self.drop_from_top, max(0.0, self.top_k_start - kmin))

                descended_enough = (kmin <= self.c["BOTTOM_ENTER_DEG"]) or (self.drop_from_top >= self.c["ROM_DROP_MIN"])
                turning_up = vel_k > +5.0
                nearly_still = abs(vel_k) < 6.0

                if descended_enough and (turning_up or nearly_still):
                    des_time = (now - self._des_start) if self._des_start else None
                    if des_time and des_time > 1e-3 and self._knee_at_top is not None:
                        des_angle = max(self._knee_at_top - kmin, 0.0)
                        self.last_down_vel = des_angle / max(des_time, 1e-3)
                        self.down_vels.append(self.last_down_vel)
                    self.t_bottom = now
                    goto("BOTTOM"); msg="Bottom (hold)"

            elif self.state=="BOTTOM":
                if (now - self.t_bottom) >= c["require_bottom_hold_s"]:
                    if kmin > (c["BOTTOM_ENTER_DEG"] + c["BOTTOM_EXIT_DELTA"]):
                        self._asc_start = now
                        self._knee_at_bottom = kmin
                        self._up_norm_sum = 0.0; self._up_norm_n = 0
                        goto("ASCENT"); msg="Ascent"
                else:
                    msg="Bottom (hold)"

            elif self.state=="ASCENT":
                if self.up_vel_norm_frame > 0:
                    self._up_norm_sum += self.up_vel_norm_frame
                    self._up_norm_n += 1

                # Easier near-top completion
                if kmin >= c["knee_top_deg"] - 8:
                    cycle = now - self.t_rep
                    up_time = (now - self._asc_start) if self._asc_start else None

                    if up_time and up_time > 1e-3 and self._knee_at_bottom is not None:
                        up_angle = max(c["knee_top_deg"] - self._knee_at_bottom, 0.0)
                        self.last_up_vel = up_angle / max(up_time, 1e-3)
                        self.up_vels.append(self.last_up_vel)

                    valid = True
                    if cycle < c["min_cycle_s"]: valid=False
                    if up_time is not None and up_time < c["min_up_s"]: valid=False
                    if self._des_start is not None and (self.t_bottom - self._des_start) < c["min_down_s"]: valid=False

                    if valid:
                        self.rep += 1
                        self.reps_in_set += 1
                        self.t_rep = now
                        self.last_top_time = now
                        self.last_cycle = cycle
                        if self._rom_min_curr is not None:
                            self.last_depth = self._rom_min_curr
                            self.rom_mins.append(self._rom_min_curr)
                        self.rep_times.append(cycle)

                        self._idle_history.append(self.idle_top_s)
                        if self._rom_min_curr is not None:
                            self._depth_history.append(self._rom_min_curr)

                        if getattr(self, "_up_norm_n", 0) > 0:
                            self.up_vel_norm_rep = self._up_norm_sum / self._up_norm_n
                        else:
                            self.up_vel_norm_rep = max(0.0, self.last_up_vel / 180.0)

                        self._rep_up_norm_history.append(self.up_vel_norm_rep)
                        self._rep_down_history.append(self.last_down_vel)
                        self._rep_cycle_history.append(self.last_cycle)

                        last_rom = self._rom_min_curr if self._rom_min_curr is not None else kmin
                        self._update_motivation(cycle, last_rom)
                        self._rep_mot_history.append(self.motivation_score)

                        if self.motivation_label == "Motivated":
                            self.good_reps_total += 1
                            self.good_reps_in_set += 1

                        # Calories increment
                        self.calories_now += self.cal_per_rep

                        msg,col=f"Rep {self.rep}",(0,255,0)
                        self._maybe_finish_set_and_rest()
                    else:
                        msg,col="Too fast (bounce/short phase)",(0,165,255)

                    # reset -> TOP
                    self._des_start=self._asc_start=None
                    self._knee_at_top=self._knee_at_bottom=None
                    self._rom_min_curr=None
                    self.top_k_start=None; self.drop_from_top=0.0
                    goto("TOP")

        exit_after_rest = self._maybe_end_rest()
        if exit_after_rest:
            return {"__exit__": True, "rep": self.rep, "msg": "Workout complete", "col": (0,255,0)}

        if kmin <= c["knee_bottom_deg"]+8 and trunk > c["trunk_lean_max_deg"]:
            msg,col="Excess forward lean",(0,140,255)

        hint = None
        if self.live_mot < 45:
            hint = "Drive up faster"
        elif self.last_down_vel > 100:
            hint = "Control the drop"

        self.msg = msg
        self.col = col

        now_ts = time.time()
        if not self.in_rest:
            self.frame_idx += 1
            if self.frame_idx % self.c["LOG_EVERY_N_FRAMES"] == 0:
                append_csv(self.c["CSV_PATH"], [
                    f"{now_ts:.3f}", self.set_idx, self.reps_in_set, self.state,
                    f"{self.up_vel_norm_frame:.4f}", f"{self.cadence_live:.2f}",
                    f"{self.last_down_vel:.2f}", f"{self.last_up_vel:.2f}",
                    "", "", "",
                    f"{self.live_mot:.1f}", self.live_label, "", ""
                ])

        return {
            "rep": self.rep,
            "msg": self.msg,
            "col": self.col,
            "down_vel": float(self.last_down_vel),
            "up_vel": float(self.last_up_vel),
            "cycle": float(self.last_cycle),
            "depth": float(self.last_depth),
            "mot_score": int(self.motivation_score),
            "mot_label": self.motivation_label,
            "live_mot": float(self.live_mot),
            "live_label": self.live_label,
            "hint": hint,
            "rest_left": float(max(0.0, (self.rest_end_time - time.time()))) if self.in_rest and self.rest_end_time else 0.0,
            "workout_done": self.workout_done,
            "calories_now": float(self.calories_now)
        }

# ---------------- Main ----------------
def main():
    print("[STEP] Opening camera…")
    cap, idx, mode = try_open()
    if cap is None:
        print("[ERROR] Cannot open camera. Enable camera for desktop apps and close Zoom/Discord/Teams.")
        return
    print(f"[OK] Camera opened on index {idx} ({mode}).")

    # Boost webcam FPS & reduce buffering
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, CFG["TARGET_FPS"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["CAM_W"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["CAM_H"])
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # Start threaded reader to avoid dropped/old frames
    grabber = FrameGrabber(cap).start()

    # --------- TTS init + start phrases ----------
    tts = TTSWorker(CFG)
    START_TIPS = [
        "Make sure to keep your back straight and lean a bit forward.",
        "Brace your core, keep your chest proud, and sit between your hips.",
        "Keep your heels down, track your knees over the mid foot, and control the drop."
    ]


    seen_body_once = False
    start_prompt_spoken = False

    # --- Rest speech control ---
    rest_countdown_started = False
    def start_breathe_countdown():
        def _runner():
            tts.say("Time to breathe.")
            for i in range(12, 0, -1):
                tts.say(str(i)); time.sleep(1.0)
            tts.say("Three, two, one, go.")
        threading.Thread(target=_runner, daemon=True).start()

    workout_summary_spoken = False
    pending_exit_from_update = False

    print("[STEP] Importing MediaPipe Pose…")
    try:
        from mediapipe import solutions as mp_solutions
        mp_pose = mp_solutions.pose
        mp_drawing = mp_solutions.drawing_utils
        mp_styles = mp_solutions.drawing_styles
    except Exception:
        import traceback
        print("[FATAL] MediaPipe import failed.")
        traceback.print_exc()
        print("Try: pip install --upgrade mediapipe==0.10.14 protobuf")
        grabber.stop(); cap.release()
        return

    def lm_xy(lms, idx, w, h):
        lm=lms[idx]; return (int(lm.x*w), int(lm.y*h))

    def get_pts(lms, w, h):
        L=mp_pose.PoseLandmark
        idxs={"l_hip":L.LEFT_HIP,"r_hip":L.RIGHT_HIP,"l_knee":L.LEFT_KNEE,"r_knee":L.RIGHT_KNEE,
              "l_ankle":L.LEFT_ANKLE,"r_ankle":L.RIGHT_ANKLE,"l_shoulder":L.LEFT_SHOULDER,"r_shoulder":L.RIGHT_SHOULDER}
        return {k: lm_xy(lms, int(v), w, h) for k,v in idxs.items()}

    coach = SquatCoach(CFG)

    print(f"[STEP] Loading pose model (fast)…")
    try:
        pose = mp_pose.Pose(
            model_complexity=0,
            smooth_landmarks=False,  # less lag = more sensitive
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception:
        import traceback
        print("[FATAL] Pose() init failed.")
        traceback.print_exc()
        grabber.stop(); cap.release()
        return

    # Window setup: resizable + fullscreen toggle
    WIN = "Squat Coach"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CFG["CAM_W"], CFG["CAM_H"])
    fullscreen = False

    hud_mode = CFG.get("HUD_MODE", "mini")  # "mini" | "full" | "off"
    hud_modes = ["mini", "full", "off"]

    print(f"[READY] {CFG['NUM_SETS']} sets × {CFG['SET_SIZE']} reps | Rest = {CFG['REST_SEC']}s | H: HUD  F: Fullscreen  Q: Quit")
    frame_count = 0
    try:
        while True:
            frame, ts = grabber.read()
            if frame is None:
                time.sleep(0.001); continue

            frame = cv2.flip(frame, 1)
            h,w = frame.shape[:2]

            # downscale before pose to reduce compute
            target_w = CFG["POSE_INPUT_W"]
            target_h = int(target_w * h / max(1, w))
            frame_small = cv2.resize(frame, (target_w, target_h))
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            res = pose.process(rgb_small)

            draw_now = (CFG["DRAW_EVERY_N"] == 1) or ((frame_count % CFG["DRAW_EVERY_N"]) == 0)
            frame_count += 1

            body_ok = False
            vis_frac = 0.0
            info = None

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                vis_frac = percent_visible(lms, CFG["vis_conf_thresh"])
                L = mp_pose.PoseLandmark
                required = [L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE,
                            L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER]
                keys_ok = joints_visible(lms, required, CFG["vis_conf_thresh"]) if CFG["require_key_joints"] else True
                body_ok = (vis_frac >= CFG["min_visible_pct"]) and keys_ok

                if body_ok and not seen_body_once:
                    seen_body_once = True
                    if not start_prompt_spoken:
                        try: tts.wait_empty(timeout_sec=0.2)
                        except Exception: pass
                        tip = random.choice(START_TIPS)
                        tts.say(f"Okay, ready to go. Three, two, one, start. {tip}")
                        start_prompt_spoken = True

                if body_ok:
                    # Re-project landmarks to full-res for drawing
                    res_full = res
                    res_full.pose_landmarks  # exists
                    # Draw with slimmer spec so you can see yourself clearly
                    if draw_now:
                        # Custom thin spec
                        landmark_spec = mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2)
                        connection_spec = mp_drawing.DrawingSpec(color=(0,170,255), thickness=1, circle_radius=2)
                        # We need to scale the landmarks back to full frame via connections util:
                        mp_drawing.draw_landmarks(
                            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=landmark_spec,
                            connection_drawing_spec=connection_spec
                        )

                    pts = get_pts(lms, w, h)
                    info = coach.update(pts)
                    if "__exit__" in (info or {}):
                        pending_exit_from_update = True
                else:
                    # unobtrusive hint (small, no big black bar)
                    cv2.putText(frame, f"Need ~50% visible (now {int(vis_frac*100)}%)",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(frame,"No person detected",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ---------- ENTERING REST ----------
            if coach.in_rest and not rest_countdown_started:
                start_breathe_countdown()
                rest_countdown_started = True
            if not coach.in_rest and rest_countdown_started:
                rest_countdown_started = False

            # ---------- HUD ----------
            if info is not None:
                rest_left = info.get("rest_left", 0.0)
                workout_done = info.get("workout_done", False)
                draw_hud(
                    frame, info, coach.set_idx, CFG["NUM_SETS"],
                    rest_left=rest_left, workout_done=workout_done,
                    good_in_set=coach.good_reps_in_set, total_in_set=coach.reps_in_set,
                    mode=hud_mode, alpha=CFG["HUD_ALPHA"], side_w=CFG["HUD_SIDEBAR_W"]
                )

            # Workout finished? Speak summary once, then quit.
            if (coach.workout_done or pending_exit_from_update) and not workout_summary_spoken:
                workout_summary_spoken = True
                if coach.last_llm_workout_text:
                    tts.say("Workout summary.")
                    tts.say(coach.last_llm_workout_text)
                else:
                    tts.say("Workout complete.")
                tts.wait_empty(timeout_sec=30)
                time.sleep(0.1)
                break

            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('h'):
                i = hud_modes.index(hud_mode) if hud_mode in hud_modes else 0
                hud_mode = hud_modes[(i+1) % len(hud_modes)]
                print(f"[HUD] mode -> {hud_mode}")
            elif k == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    finally:
        try:
            try: tts.wait_empty(timeout_sec=5)
            except Exception: pass
            if 'pose' in locals() and pose is not None:
                try: pose.close()
                except Exception: pass
        finally:
            grabber.stop(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[FATAL] Top-level crash:")
        traceback.print_exc()
