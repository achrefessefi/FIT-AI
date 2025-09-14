# push.py
import cv2, time, numpy as np, os, json
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # so you can set GROQ_API_KEY, GROQ_ENABLE, GROQ_MODEL in .env

# ---------------- Camera open (Windows-friendly) ----------------
def try_open():
    for flag in [cv2.CAP_DSHOW, 0]:
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, flag) if flag else cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap
            cap.release()
    return None

# ---------------- Helpers ----------------
def angle_deg(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    v1=np.array([ax-bx, ay-by], np.float32); v2=np.array([cx-bx, cy-by], np.float32)
    n1=np.linalg.norm(v1)+1e-6; n2=np.linalg.norm(v2)+1e-6
    cos=np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
    return float(np.degrees(np.arccos(cos)))

def point_line_distance_ratio(p, a, b):
    """Hip deviation from shoulder-ankle line, normalized by line length (0..1+)."""
    ax,ay = a; bx,by = b; px,py = p
    denom = np.hypot(bx-ax, by-ay) + 1e-6
    num = abs((by-ay)*px - (bx-ax)*py + bx*ay - by*ax) / denom
    return num / denom

class EMA:
    def __init__(self,a): self.a=a; self.y=None
    def push(self,x): self.y=x if self.y is None else self.y+(x-self.y)*self.a; return self.y

class Vel:
    def __init__(self): self.x=None; self.t=None
    def update(self,x):
        now=time.time()
        if self.x is None:
            self.x, self.t = x, now
            return 0.0
        dt=max(1e-3, now-self.t)
        v=(x-self.x)/dt
        self.x, self.t = x, now
        return v

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---------------- Config (form & speed thresholds) ----------------
CFG = {
    "elbow_top_deg": 165,      # realistic near-lockout
    "elbow_bottom_deg": 95,    # realistic depth target (~90-100)
    "ema_alpha": 0.6,          # smoothing
    "vis_thr": 0.35,           # minimal visibility per joint for chosen side
    "v_down_start": -5,
    "v_up_start": 5,
    # --- Groq ---
    "GROQ_ENABLE": (os.getenv("GROQ_ENABLE", "1").strip().lower() in ("1","true","yes","on")),
    "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
}

# Speed/rapidity logic
CFG_SPEED = {
    "rapid_ttt_sec": 1.60,
    "energetic_up_peak": 120,
    "energetic_rom": 60,       # ROM considered solid (deg)
    "energetic_rpm": 25,
    "rpm_ema_alpha": 0.3,
    "history_max": 50
}

# Workout protocol (5-15-5-15-5)
# Workout protocol (env-driven like squat.py)
def _env_int(key, default):
    try:
        v = os.getenv(key)
        return int(v) if v is not None else default
    except Exception:
        return default

CFG_WORKOUT = {
    "reps_per_set": _env_int("SET_SIZE", 10),  # keep name aligned with squat.py
    "num_sets": _env_int("NUM_SETS", 3),
    "rest_sec": _env_int("REST_SEC", 15),
}

# UI ranges / thresholds
CFG_UI = {
    "rep_duration_range": (0.6, 4.0),
    "rep_good": 1.8, "rep_bad": 3.2,
    # kept for motivation only
    "rest_range": (0.0, 6.0),
    "rest_good": 1.5, "rest_bad": 4.0,
    "w_rep": 0.6, "w_rest": 0.4,
    "bonus_energetic": 8, "penalty_shallow": 6,
}

# ---- Realistic scoring knobs ----
SCORE_WTS = {   # weights sum ~100
    "rom": 30,
    "bottom": 15,
    "lockout": 10,
    "plank": 25,
    "tempo": 10,
    "smooth": 10
}

SCORE_CFG = {
    "rom_good": 70.0,   # deg full credit
    "rom_min":  35.0,   # deg heavy penalty

    "bottom_full_penalty_at_deg": 20.0,  # 20° shallow => full bottom penalty
    "lockout_full_penalty_at_deg": 20.0, # 20° short => full lockout penalty

    "plank_green": 0.04,   # <4% deviation great
    "plank_red":   0.12,   # >12% deviation poor (sag/pike)

    "ttt_opt": 1.6, "ttt_lo": 0.8, "ttt_hi": 3.0,
    "tempo_balance_full_penalty_at_ratio_diff": 0.6,  # >=60% diff => full penalty

    "jerk_green": 60.0, "jerk_red": 220.0,  # deg/s^2
}

# ---------------- Color & UI helpers ----------------
def color_for_value_sec(x, good_thr, bad_thr):
    if x is None: return (128,128,128)
    if x <= good_thr: return (60,180,60)
    if x >= bad_thr:  return (40,40,220)
    return (0,215,255)

def draw_hbar(frame, x, y, w, h, value, vrange, label, color, text_color=(255,255,255), invert=False):
    vmin, vmax = vrange
    if value is None:
        frac = 0.0
        val_txt = "—"
    else:
        frac = clamp((value - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        if invert: frac = 1.0 - frac
        val_txt = f"{value:.2f}s"
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*frac), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (200,200,200), 1)
    cv2.putText(frame, f"{label}: {val_txt}", (x+8, y+h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

def draw_motivation_badge(frame, x, y, score):
    txt = f"Motivation: {score if score is not None else 0}"
    col = (60,180,60) if (score is not None and score >= 70) else \
          (0,215,255) if (score is not None and score >= 40) else \
          (40,40,220)
    cv2.rectangle(frame, (x, y), (x+220, y+36), (30,30,30), -1)
    cv2.putText(frame, txt, (x+10, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

# ---------------- Interpretations (console text) ----------------
def label_from_score(s):
    if s is None: return "—"
    if s >= 90: return "Excellent"
    if s >= 75: return "Good"
    if s >= 60: return "Fair"
    return "Needs work"

def interp_rom(rom):
    if rom is None: return "ROM unknown"
    if rom >= SCORE_CFG["rom_good"]: return f"full depth ({rom:.0f}°)"
    if rom >= 50: return f"solid depth ({rom:.0f}°)"
    if rom >= 40: return f"okay depth ({rom:.0f}°)"
    return f"shallow ({rom:.0f}°)"

def interp_bottom(min_angle):
    tgt = CFG["elbow_bottom_deg"]
    if min_angle is None: return "bottom unknown"
    if min_angle <= tgt + 5: return "bottom reached"
    return f"bottom shallow by {min_angle - tgt:.0f}°"

def interp_lockout(top_end_angle):
    tgt = CFG["elbow_top_deg"]
    if top_end_angle is None: return "lockout unknown"
    if top_end_angle >= tgt - 5: return "locked out"
    return f"short of lockout by {tgt - top_end_angle:.0f}°"

def interp_plank(dev):
    if dev is None: return "plank unknown"
    if dev <= SCORE_CFG["plank_green"]: return "plank solid"
    if dev <= (SCORE_CFG["plank_green"] + SCORE_CFG["plank_red"])/2: return "minor hip sag/pike"
    return "significant hip sag/pike"

def interp_tempo(ttt, asc, desc):
    parts = []
    sc = SCORE_CFG
    if ttt is None:
        parts.append("tempo unknown")
    else:
        if sc["ttt_lo"] < ttt < sc["ttt_hi"]:
            parts.append(f"cadence ok ({ttt:.2f}s)")
        elif ttt <= sc["ttt_lo"]:\

            parts.append(f"too fast ({ttt:.2f}s)")
        else:
            parts.append(f"too slow ({ttt:.2f}s)")
    if asc and desc and asc > 0 and desc > 0:
        diff = abs(asc - desc) / max(asc, desc)
        if diff < 0.2: parts.append("balanced up/down")
        elif diff < 0.4: parts.append("slightly unbalanced")
        else: parts.append("unbalanced")
    return ", ".join(parts)

def interp_smooth(jerk_std):
    if jerk_std is None: return "smoothness unknown"
    if jerk_std <= SCORE_CFG["jerk_green"]: return "very smooth"
    if jerk_std <= (SCORE_CFG["jerk_green"] + SCORE_CFG["jerk_red"])/2: return "moderately smooth"
    return "jerky"

# --------- Coach notes & cues (STATIC RULES) ----------
def verdict_from_score(avg):
    if   avg is None: return "Unknown"
    if   avg >= 80:   return "Good"
    elif avg >= 65:   return "Medium"
    elif avg >= 50:   return "Could do better"
    else:             return "Poor"

def pushup_notes_from_aggregates(avg_ttt, avg_rom, avg_rpm, med_plank, avg_jerk, lockout_rate, bottom_rate, bal_diff_ratio):
    notes = []
    # Tempo / cadence
    if avg_ttt is not None:
        if avg_ttt < SCORE_CFG["ttt_lo"]:
            notes.append("tempo_too_fast")
        elif avg_ttt > SCORE_CFG["ttt_hi"]:
            notes.append("tempo_too_slow")
        else:
            notes.append("tempo_ok")
    if avg_rpm is not None:
        if avg_rpm >= 25: notes.append("cadence_high")
        elif avg_rpm >= 15: notes.append("cadence_moderate")
        else: notes.append("cadence_low")

    # ROM / depth
    if avg_rom is not None:
        if avg_rom >= SCORE_CFG["rom_good"]:
            notes.append("depth_full")
        elif avg_rom >= 50:
            notes.append("depth_solid")
        elif avg_rom >= 40:
            notes.append("depth_okay")
        else:
            notes.append("depth_shallow")

    # Lockout & bottom consistency
    if lockout_rate is not None:
        if lockout_rate >= 0.9: notes.append("lockout_consistent")
        elif lockout_rate >= 0.7: notes.append("lockout_most")
        else: notes.append("lockout_spotty")
    if bottom_rate is not None:
        if bottom_rate >= 0.9: notes.append("bottom_consistent")
        elif bottom_rate >= 0.7: notes.append("bottom_most")
        else: notes.append("bottom_shallow_often")

    # Plank
    if med_plank is not None:
        if med_plank <= SCORE_CFG["plank_green"]: notes.append("plank_strong")
        elif med_plank <= (SCORE_CFG["plank_green"]+SCORE_CFG["plank_red"])/2: notes.append("plank_minor_sag")
        else: notes.append("plank_sag")

    # Smoothness
    if avg_jerk is not None:
        if avg_jerk <= SCORE_CFG["jerk_green"]: notes.append("smooth_very")
        elif avg_jerk <= (SCORE_CFG["jerk_green"]+SCORE_CFG["jerk_red"])/2: notes.append("smooth_ok")
        else: notes.append("smooth_jerky")

    # Up/Down balance
    if bal_diff_ratio is not None:
        if bal_diff_ratio < 0.2: notes.append("balance_even")
        elif bal_diff_ratio < 0.4: notes.append("balance_slight")
        else: notes.append("balance_unbalanced")

    if not notes:
        notes.append("general_ok")
    return notes

def pushup_cues_from_notes(notes):
    cues = []
    # safety & form first
    if "plank_sag" in notes:
        cues.append("Squeeze glutes and brace your core—keep a straight line from shoulders to ankles.")
    elif "plank_minor_sag" in notes:
        cues.append("Lightly tuck your pelvis and keep ribs down to avoid hip sag.")
    if "depth_shallow" in notes or "bottom_shallow_often" in notes:
        cues.append("Aim your chest closer to the floor—touch a consistent bottom position.")
    if "lockout_spotty" in notes:
        cues.append("Finish every rep—press until your elbows reach full lockout.")
    if "smooth_jerky" in notes:
        cues.append("Move continuously—avoid pausing mid-range or bouncing at the bottom.")
    if "tempo_too_fast" in notes:
        cues.append("Slow the cadence slightly—control down, then drive up.")
    if "tempo_too_slow" in notes:
        cues.append("Tighten the rhythm—start the next rep sooner at the top.")
    if not cues:
        cues.append("Great work—keep the cadence steady and the body line tight.")
    return cues[:3]

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def groq_send(kind, payload):
    if not CFG.get("GROQ_ENABLE", False):
        return None
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[GROQ] GROQ_API_KEY not set; skipping.")
            return None
        client = Groq(api_key=api_key)
        model = CFG.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        system = (
            "You are an energetic, professional calisthenics coach.\n"
            "Given push-up metrics JSON, write 2–4 SHORT sentences that:\n"
            "• Celebrate wins with numbers.\n"
            "• Give 1–2 precise cues based on the provided notes.\n"
            "• End with a motivating call-to-action.\n"
            "STYLE: concise, varied, positive; safety first."
        )
        header = "Athlete finished a set/workout. Respond with a tight, powerful summary.\n\n"
        user = header + json.dumps(payload, ensure_ascii=False)
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user},
            ],
            temperature=0.8,
            max_tokens=280,
        )
        text = rsp.choices[0].message.content.strip()
        tag = "SET" if kind=="set" else "WORKOUT"
        print(f"=== GROQ {tag} ===\n{text}\n=== END ===")
        return text
    except Exception as e:
        print("[GROQ] Error:", repr(e))
        return None

# ---------------- Side chain picker (includes hip & ankle) ----------------
def pick_side_pts(lms, w, h, mp_pose, vis_thr):
    L = mp_pose.PoseLandmark
    def side_ok(side):
        if side=="left":
            ids = [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP, L.LEFT_ANKLE]
        else:
            ids = [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP, L.RIGHT_ANKLE]
        if any(lms[int(i)].visibility < vis_thr for i in ids): return False, None
        def xy(i):
            lm=lms[int(i)]; return (int(lm.x*w), int(lm.y*h))
        if side=="left":
            return True, {
                "shoulder": xy(L.LEFT_SHOULDER),
                "elbow":    xy(L.LEFT_ELBOW),
                "wrist":    xy(L.LEFT_WRIST),
                "hip":      xy(L.LEFT_HIP),
                "ankle":    xy(L.LEFT_ANKLE),
            }
        else:
            return True, {
                "shoulder": xy(L.RIGHT_SHOULDER),
                "elbow":    xy(L.RIGHT_ELBOW),
                "wrist":    xy(L.RIGHT_WRIST),
                "hip":      xy(L.RIGHT_HIP),
                "ankle":    xy(L.RIGHT_ANKLE),
            }
    okL, ptsL = side_ok("left")
    okR, ptsR = side_ok("right")
    if okL and okR:
        def avg_vis(ids): return sum(lms[int(i)].visibility for i in ids)/len(ids)
        Lids=[L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP, L.LEFT_ANKLE]
        Rids=[L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP, L.RIGHT_ANKLE]
        return (ptsL, "left") if avg_vis(Lids)>=avg_vis(Rids) else (ptsR, "right")
    if okL: return ptsL, "left"
    if okR: return ptsR, "right"
    return None, None

# ---------------- Push-up Counter + Scoring ----------------
class PushupCounter:
    # States: TOP -> DESCENT -> BOTTOM -> ASCENT -> TOP
    def __init__(self, cfg, cfg_speed):
        self.c = cfg
        self.s = cfg_speed
        self.state = "TOP"
        self.reps = 0
        self.ema = EMA(cfg["ema_alpha"])
        self.vel = Vel()
        self.allow_count = True

        # timing/metrics
        self.last_top_t = None
        self.pending_rest_gap = None
        self.desc_start_t = None
        self.bottom_t = None
        self.asc_start_t = None

        # kinematics
        self.min_elbow_angle = None
        self.top_elbow_at_end = None

        # smoothness & plank
        self.v_samples = []          # velocities during the rep
        self.max_plank_dev = 0.0     # max hip deviation ratio in the rep

        # UI / outputs
        self.rpm_ema = EMA(self.s["rpm_ema_alpha"])
        self.rep_history = deque(maxlen=self.s["history_max"])

        self.last_rep_duration = None
        self.last_rest_gap = None
        self.last_motivation = None
        self.last_form_score = None

        self.cur_pts = None

    def set_enabled(self, enabled: bool):
        self.allow_count = enabled

    def _start_descent(self, now, elbow_at_start):
        if self.last_top_t is not None:
            self.pending_rest_gap = max(0.0, now - self.last_top_t)
        else:
            self.pending_rest_gap = None

        self.desc_start_t = now
        self.bottom_t = None
        self.asc_start_t = None

        self.min_elbow_angle = 999.0
        self.v_samples = []
        self.max_plank_dev = 0.0

    # ---------- Motivation ----------
    def _motivation(self, ttt, rest, energetic, rom, cfg_ui):
        lo_rep, hi_rep = cfg_ui["rep_duration_range"]
        lo_rest, hi_rest = cfg_ui["rest_range"]
        def norm01(x, lo, hi):
            if x is None: return 0.0
            if hi <= lo: return 0.0
            return clamp((x - lo) / (hi - lo), 0.0, 1.0)
        n_rep = norm01(ttt, lo_rep, hi_rep)
        n_rest = norm01(rest, lo_rest, hi_rest)
        base = 100.0 - (100.0 * (cfg_ui["w_rep"]*n_rep + cfg_ui["w_rest"]*n_rest))
        if energetic: base += cfg_ui["bonus_energetic"]
        if (rom is not None) and (rom < CFG_SPEED["energetic_rom"]): base -= cfg_ui["penalty_shallow"]
        return int(clamp(base, 0, 100))

    # ---------- Realistic form scoring ----------
    def _form_score(self, rom, min_angle, top_end_angle, ascent_time, descent_time, jerk_std, plank_dev_max, ttt):
        w = SCORE_WTS; sc = SCORE_CFG
        score = 0.0
        total_w = sum(w.values()) + 1e-6

        # ROM
        if rom is None: rom = 0.0
        rom_frac = clamp((rom - sc["rom_min"]) / max(1.0, sc["rom_good"] - sc["rom_min"]), 0.0, 1.0)
        score += w["rom"] * rom_frac

        # Bottom depth
        if min_angle is None: min_angle = 999.0
        bottom_deficit = max(0.0, min_angle - self.c["elbow_bottom_deg"])
        bottom_frac = 1.0 - clamp(bottom_deficit / sc["bottom_full_penalty_at_deg"], 0.0, 1.0)
        score += w["bottom"] * bottom_frac

        # Lockout
        if top_end_angle is None: top_end_angle = 0.0
        lockout_deficit = max(0.0, self.c["elbow_top_deg"] - top_end_angle)
        lockout_frac = 1.0 - clamp(lockout_deficit / sc["lockout_full_penalty_at_deg"], 0.0, 1.0)
        score += w["lockout"] * lockout_frac

        # Plank
        pd = plank_dev_max if plank_dev_max is not None else 1.0
        if pd <= sc["plank_green"]:
            plank_frac = 1.0
        elif pd >= sc["plank_red"]:
            plank_frac = 0.0
        else:
            plank_frac = 1.0 - (pd - sc["plank_green"]) / (sc["plank_red"] - sc["plank_green"])
        score += w["plank"] * clamp(plank_frac, 0.0, 1.0)

        # Tempo
        if ttt is None: ttt = sc["ttt_opt"]
        if ttt <= sc["ttt_lo"] or ttt >= sc["ttt_hi"]:
            ttt_frac = 0.0
        else:
            if ttt <= sc["ttt_opt"]:
                ttt_frac = (ttt - sc["ttt_lo"]) / (sc["ttt_opt"] - sc["ttt_lo"])
            else:
                ttt_frac = (sc["ttt_hi"] - ttt) / (sc["ttt_hi"] - sc["ttt_opt"])
            ttt_frac = clamp(ttt_frac, 0.0, 1.0)

        if ascent_time and descent_time and ascent_time > 0 and descent_time > 0:
            diff_ratio = abs(ascent_time - descent_time) / max(ascent_time, descent_time)
            bal_frac = 1.0 - clamp(diff_ratio / sc["tempo_balance_full_penalty_at_ratio_diff"], 0.0, 1.0)
        else:
            bal_frac = 1.0
        tempo_frac = 0.6*ttt_frac + 0.4*bal_frac
        score += w["tempo"] * clamp(tempo_frac, 0.0, 1.0)

        # Smoothness
        j = jerk_std if jerk_std is not None else sc["jerk_red"]
        if j <= sc["jerk_green"]:
            smooth_frac = 1.0
        elif j >= sc["jerk_red"]:
            smooth_frac = 0.0
        else:
            smooth_frac = 1.0 - (j - sc["jerk_green"]) / (sc["jerk_red"] - sc["jerk_green"])
        score += w["smooth"] * clamp(smooth_frac, 0.0, 1.0)

        return int(round(100.0 * score / total_w))

    def _finish_rep(self, now, elbow_end_angle, ttt):
        ascent_time = (now - self.asc_start_t) if self.asc_start_t else None
        descent_time = (self.bottom_t - self.desc_start_t) if (self.bottom_t and self.desc_start_t) else None

        # ROM
        rom = max(0.0, self.c["elbow_top_deg"] - (self.min_elbow_angle if self.min_elbow_angle is not None else self.c["elbow_top_deg"]))

        # Cadence
        rpm_inst = (60.0 / ttt) if ttt and ttt > 1e-3 else 0.0
        rpm_smoothed = self.rpm_ema.push(rpm_inst) if rpm_inst > 0 else (self.rpm_ema.y or 0.0)

        # Smoothness (jerk = std of dv)
        jerk_std = None
        if len(self.v_samples) >= 3:
            dv = np.diff(self.v_samples)
            jerk_std = float(np.std(dv))

        # Motivation
        energetic = (rom >= self.s["energetic_rom"]) and (rpm_smoothed is not None and rpm_smoothed >= self.s["energetic_rpm"])
        motivation = self._motivation(ttt, self.pending_rest_gap, energetic, rom, CFG_UI)

        # Form score
        form_score = self._form_score(
            rom=rom,
            min_angle=self.min_elbow_angle,
            top_end_angle=elbow_end_angle,
            ascent_time=ascent_time,
            descent_time=descent_time,
            jerk_std=jerk_std,
            plank_dev_max=self.max_plank_dev,
            ttt=ttt
        )

        rep_info = {
            "rep": self.reps,
            "time": now,
            "top_to_top_s": round(ttt, 3) if ttt else None,
            "rest_gap_s": round(self.pending_rest_gap, 3) if self.pending_rest_gap is not None else None,
            "rpm_inst": round(rpm_inst, 1),
            "rpm_ema": round(rpm_smoothed or 0.0, 1),
            "ascent_s": round(ascent_time, 3) if ascent_time else None,
            "descent_s": round(descent_time, 3) if descent_time else None,
            "rom_deg": round(rom, 1),
            "max_plank_dev": round(self.max_plank_dev, 3),
            "jerk_std": None if jerk_std is None else int(jerk_std),
            "form_score": form_score,
            "motivation": motivation,
            "min_elbow_angle": None if self.min_elbow_angle is None else round(self.min_elbow_angle,1),
            "top_end_angle": None if elbow_end_angle is None else round(elbow_end_angle,1),
            # convenience booleans
            "hit_lockout": (True if (elbow_end_angle is not None and elbow_end_angle >= self.c["elbow_top_deg"]-5) else False),
            "hit_bottom": (True if (self.min_elbow_angle is not None and self.min_elbow_angle <= self.c["elbow_bottom_deg"]+5) else False),
            "balance_diff_ratio": (abs(ascent_time - descent_time)/max(ascent_time, descent_time) if (ascent_time and descent_time and ascent_time>0 and descent_time>0) else None),
        }
        self.rep_history.append(rep_info)

        # HUD buffers
        self.last_rep_duration = ttt
        self.last_rest_gap = self.pending_rest_gap
        self.last_motivation = motivation
        self.last_form_score = form_score
        self.top_elbow_at_end = elbow_end_angle

        self.last_top_t = now

        # ---- Detailed console interpretation for this rep ----
        rom_txt = interp_rom(rom)
        bottom_txt = interp_bottom(self.min_elbow_angle)
        lockout_txt = interp_lockout(elbow_end_angle)
        plank_txt = interp_plank(self.max_plank_dev)
        tempo_txt = interp_tempo(ttt, ascent_time, descent_time)
        smooth_txt = interp_smooth(jerk_std)
        label = label_from_score(form_score)
        print(f"[REP {self.reps}] Score {form_score} ({label}) | ROM {rom_txt}; {bottom_txt}; {lockout_txt}; {plank_txt}; {tempo_txt}; {smooth_txt}.")
        print(f"          Metrics → TTT:{rep_info['top_to_top_s']}s  Ascent:{rep_info['ascent_s']}s  Descent:{rep_info['descent_s']}s  "
              f"PlankDev:{rep_info['max_plank_dev']}  JerkStd:{rep_info['jerk_std']}  RPMinst:{rep_info['rpm_inst']} EMA:{rep_info['rpm_ema']}  Motivation:{motivation}")

        return rep_info

    def update(self, pts):
        self.cur_pts = pts
        c = self.c
        elbow = angle_deg(pts["shoulder"], pts["elbow"], pts["wrist"])
        elbow = self.ema.push(elbow)
        v = self.vel.update(elbow)
        now = time.time()

        # Update plank deviation during active phases
        if "hip" in pts and "ankle" in pts and "shoulder" in pts:
            dev = point_line_distance_ratio(pts["hip"], pts["shoulder"], pts["ankle"])
            if self.state in ("DESCENT", "BOTTOM", "ASCENT"):
                self.max_plank_dev = max(self.max_plank_dev, float(dev))

        # Track in phases
        if self.state in ("DESCENT", "BOTTOM", "ASCENT"):
            if self.state in ("DESCENT", "BOTTOM"):
                self.min_elbow_angle = min(self.min_elbow_angle or 999.0, elbow)
            # smoothness sample
            self.v_samples.append(float(v))

        if self.state == "TOP":
            if v < c["v_down_start"] or elbow < (c["elbow_top_deg"] - 5):
                self._start_descent(now, elbow_at_start=elbow)
                self.state = "DESCENT"

        elif self.state == "DESCENT":
            deep = elbow <= c["elbow_bottom_deg"]
            slowing_or_flip = abs(v) < 15 or v > 0
            if deep and slowing_or_flip:
                self.bottom_t = now
                self.state = "BOTTOM"

        elif self.state == "BOTTOM":
            if v > c["v_up_start"] and elbow > (c["elbow_bottom_deg"] + 5):
                self.asc_start_t = now
                self.state = "ASCENT"

        elif self.state == "ASCENT":
            near_top = elbow >= (c["elbow_top_deg"] - 5)
            slowing  = abs(v) < 15 or v < 0
            if near_top and slowing:
                counted = False
                rep_info = None
                ttt = (now - self.last_top_t) if self.last_top_t is not None else None
                if self.allow_count:
                    self.reps += 1
                    counted = True
                rep_info = self._finish_rep(now, elbow_end_angle=elbow, ttt=ttt)
                self.state = "TOP"
                return elbow, v, self.state, self.reps, rep_info, counted

        return elbow, v, self.state, self.reps, None, False

# ---------------- Workout Manager (with scoring + Groq summaries) ----------------
class Workout:
    def __init__(self, reps_per_set, num_sets, rest_sec):
        self.reps_per_set = reps_per_set
        self.num_sets = num_sets
        self.rest_sec = rest_sec
        self.set_idx = 0
        self.reps_in_set = 0
        self.phase = "WORK"     # WORK | REST | DONE
        self.rest_end = None
        self.current_rep_scores = []
        self.set_scores = []
        self.last_set_score_stats = None  # (avg, std, min, max)
        # store full rep infos for the active set to build notes
        self.current_set_reps = []
        self.groq_last_set = None
        self.groq_last_workout = None

    def on_rep(self, now, rep_info=None):
        if self.phase != "WORK": return
        if rep_info is not None:
            self.current_rep_scores.append(int(rep_info["form_score"]))
            self.current_set_reps.append(rep_info)
        self.reps_in_set += 1
        if self.reps_in_set >= self.reps_per_set:
            # compute set stats
            if self.current_rep_scores:
                arr = np.array(self.current_rep_scores, dtype=float)
                avg = float(np.mean(arr)); std = float(np.std(arr))
                mn = int(np.min(arr)); mx = int(np.max(arr))
                set_score = int(round(avg))
                self.set_scores.append(set_score)
                self.last_set_score_stats = (int(round(avg)), int(round(std)), mn, mx)
                label = label_from_score(set_score)
                consistency = "very consistent" if std < 5 else ("consistent" if std < 10 else "variable")
                print(f"[SET {self.set_idx+1}] Avg {set_score} ({label}) | Spread {mn}-{mx}, SD {int(round(std))} → {consistency}.")
                # ---- Build set notes/cues & call Groq
                set_payload = self._build_set_payload(self.set_idx+1, set_score)
                print("=== LLM_PROMPT:SET ===")
                print(json.dumps(set_payload, ensure_ascii=False))
                print("=== END ===")
                self.groq_last_set = groq_send("set", set_payload)
            else:
                self.set_scores.append(0)
                self.last_set_score_stats = (0, 0, 0, 0)
                print(f"[SET {self.set_idx+1}] No reps captured.")
            # transition
            self.current_rep_scores = []
            self.current_set_reps = []
            if self.set_idx < self.num_sets - 1:
                self.phase = "REST"
                self.rest_end = now + self.rest_sec
                print(f"[REST] {self.rest_sec}s rest. Prepare for Set {self.set_idx+2}.")
            else:
                self.phase = "DONE"
                print(f"[WORKOUT] All sets done. Final Score: {self.final_score() or 0} ({label_from_score(self.final_score() or 0)}).")
                # Workout payload + Groq
                w_payload = self._build_workout_payload()
                print("=== LLM_PROMPT:WORKOUT ===")
                print(json.dumps(w_payload, ensure_ascii=False))
                print("=== END ===")
                self.groq_last_workout = groq_send("workout", w_payload)

    def _aggregates_from_set_reps(self, reps):
        if not reps: return {}
        # helpers that ignore None
        def arr(key): 
            vals = [r.get(key) for r in reps if r.get(key) is not None]
            return np.array(vals, dtype=float) if vals else None
        ttt = arr("top_to_top_s"); rom = arr("rom_deg"); rpm = arr("rpm_ema")
        plank = arr("max_plank_dev"); jerk = arr("jerk_std")
        asc = arr("ascent_s"); des = arr("descent_s")
        bal = [r.get("balance_diff_ratio") for r in reps if r.get("balance_diff_ratio") is not None]
        lockouts = [1.0 for r in reps if r.get("hit_lockout")] ; locks_rate = (sum(lockouts)/len(reps)) if reps else 0.0
        bottoms = [1.0 for r in reps if r.get("hit_bottom")]   ; bot_rate  = (sum(bottoms)/len(reps)) if reps else 0.0
        return {
            "avg_ttt": float(np.mean(ttt)) if ttt is not None else None,
            "avg_rom": float(np.mean(rom)) if rom is not None else None,
            "avg_rpm": float(np.mean(rpm)) if rpm is not None else None,
            "med_plank": float(np.median(plank)) if plank is not None else None,
            "avg_jerk": float(np.mean(jerk)) if jerk is not None else None,
            "lockout_rate": locks_rate,
            "bottom_rate": bot_rate,
            "bal_diff_ratio": (float(np.mean(bal)) if bal else None)
        }

    def _build_set_payload(self, set_idx1, set_score):
        ag = self._aggregates_from_set_reps(self.current_set_reps)
        notes = pushup_notes_from_aggregates(
            ag.get("avg_ttt"), ag.get("avg_rom"), ag.get("avg_rpm"),
            ag.get("med_plank"), ag.get("avg_jerk"),
            ag.get("lockout_rate"), ag.get("bottom_rate"),
            ag.get("bal_diff_ratio")
        )
        cues = pushup_cues_from_notes(notes)
        payload = {
            "type": "set_summary",
            "timestamp_utc": now_iso(),
            "set_index": set_idx1,
            "session": {"target_sets": self.num_sets, "target_reps_per_set": self.reps_per_set},
            "metrics": {
                "avg_ttt_s": round(ag.get("avg_ttt"), 3) if ag.get("avg_ttt") is not None else None,
                "avg_rom_deg": round(ag.get("avg_rom"), 1) if ag.get("avg_rom") is not None else None,
                "avg_rpm": round(ag.get("avg_rpm"), 1) if ag.get("avg_rpm") is not None else None,
                "med_plank_dev": round(ag.get("med_plank"), 3) if ag.get("med_plank") is not None else None,
                "avg_jerk_std": None if ag.get("avg_jerk") is None else int(ag.get("avg_jerk")),
                "lockout_rate": round(ag.get("lockout_rate"), 2) if ag.get("lockout_rate") is not None else None,
                "bottom_rate": round(ag.get("bottom_rate"), 2) if ag.get("bottom_rate") is not None else None,
                "avg_balance_diff_ratio": round(ag.get("bal_diff_ratio"), 2) if ag.get("bal_diff_ratio") is not None else None
            },
            "quality": {
                "set_score_avg": set_score,
                "set_score_label": label_from_score(set_score)
            },
            "notes": notes,
            "coaching_cues": cues,
            "counts": {
                "reps_in_set": self.reps_per_set,
                "set_index": set_idx1
            }
        }
        return payload

    def _build_workout_payload(self):
        # flatten all set aggregates by averaging set scores & last set aggregates if available
        final = self.final_score()
        verdict = verdict_from_score(final)
        payload = {
            "type": "workout_summary",
            "timestamp_utc": now_iso(),
            "session": {"target_sets": self.num_sets, "target_reps_per_set": self.reps_per_set},
            "totals": {"sets_completed": self.num_sets, "reps_completed": self.num_sets*self.reps_per_set},
            "quality": {
                "final_score": final,
                "final_label": label_from_score(final),
                "set_scores": self.set_scores
            },
        }
        # try to derive big-picture notes from last set metrics we captured (closest to end-state)
        # This keeps it lightweight without re-storing every rep in every set.
        if self.last_set_score_stats:
            avg, std, mn, mx = self.last_set_score_stats
            payload["spread"] = {"avg": avg, "sd": std, "min": mn, "max": mx}
        # high-level notes: from last completed set payload if available
        # (In a richer version, you’d aggregate across sets.)
        if self.groq_last_set:
            payload["last_set_summary_text"] = self.groq_last_set
        return payload

    def tick(self, now):
        if self.phase == "REST" and self.rest_end is not None:
            if now >= self.rest_end:
                self.set_idx += 1
                self.reps_in_set = 0
                if self.set_idx >= self.num_sets:
                    self.phase = "DONE"
                else:
                    self.phase = "WORK"
                    print(f"[WORK] Set {self.set_idx+1}/{self.num_sets} starting.")

    def rest_remaining(self, now):
        if self.phase != "REST" or self.rest_end is None: return 0
        return max(0, int(self.rest_end - now))

    def headline(self):
        cur_set = min(self.set_idx+1, self.num_sets)
        cur_rep = min(self.reps_in_set, self.reps_per_set)
        return f"Set {cur_set}/{self.num_sets} • Rep {cur_rep}/{self.reps_per_set}"

    def running_set_avg(self):
        if not self.current_rep_scores: return None
        return int(sum(self.current_rep_scores)/len(self.current_rep_scores))

    def final_score(self):
        if not self.set_scores: return None
        return int(sum(self.set_scores)/len(self.set_scores))

# ---------------- Main ----------------
def main():
    cap = try_open()
    if not cap:
        print("Cannot open camera"); return

    from mediapipe import solutions as mp_solutions
    mp_pose = mp_solutions.pose
    mp_draw = mp_solutions.drawing_utils
    mp_style = mp_solutions.drawing_styles

    counter = PushupCounter(CFG, CFG_SPEED)
    workout = Workout(CFG_WORKOUT["reps_per_set"], CFG_WORKOUT["num_sets"], CFG_WORKOUT["rest_sec"])

    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("[READY] Push-up counter with realistic scoring + Groq coach summaries.")
    print("Protocol: 5 reps → 15s rest → 5 reps → 15s rest → 5 reps.")
    print("Keys: Q quit, R reset, P print last 5 reps.")
    last_status_print = 0.0

    try:
        while True:
            now = time.time()
            counter.set_enabled(workout.phase == "WORK")

            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Tick workout (rest→work transitions)
            workout.tick(now)

            hud_lines = []
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                )
                pts, side = pick_side_pts(lms, w, h, mp_pose, CFG["vis_thr"])
                if pts:
                    counter.cur_pts = pts
                    elbow, v, st, reps, rep_info, counted = counter.update(pts)

                    if counted and workout.phase == "WORK" and rep_info is not None:
                        workout.on_rep(now, rep_info=rep_info)

                    rpm = counter.rpm_ema.y or 0.0
                    hud_lines.append(workout.headline())
                    phase_txt = "WORK" if workout.phase == "WORK" else ("REST" if workout.phase == "REST" else "DONE")
                    if workout.phase == "REST":
                        hud_lines.append(f"Phase: {phase_txt}  |  Rest: {workout.rest_remaining(now)}s")
                    else:
                        hud_lines.append(f"Phase: {phase_txt}")
                    hud_lines.append(f"Reps (total): {reps}   State: {st}   Elbow:{int(elbow)}°")
                    hud_lines.append(f"RPM (EMA): {rpm:.1f}   v:{int(v)}°/s   Side:{side}")

                    rep_score_txt = f"{counter.last_form_score} /100" if counter.last_form_score is not None else "—"
                    run_set_avg = workout.running_set_avg()
                    set_avg_txt = f"{run_set_avg} /100" if run_set_avg is not None else (f"{workout.set_scores[-1]} /100" if workout.set_scores else "—")
                    final_txt = f"{workout.final_score()} /100" if workout.final_score() is not None else "—"
                    hud_lines.append(f"Scores → Rep: {rep_score_txt}   Set avg: {set_avg_txt}   Final: {final_txt}")

                else:
                    hud_lines = ["Show one side: shoulder + elbow + wrist + hip + ankle"]
            else:
                hud_lines = ["No person detected"]

            # ---- Throttled console status (1/sec) ----
            if now - last_status_print >= 1.0:
                last_status_print = now
                status_line = " | ".join([
                    workout.headline(),
                    f"Phase:{'WORK' if workout.phase=='WORK' else ('REST' if workout.phase=='REST' else 'DONE')}",
                    f"RepsTotal:{counter.reps}",
                    f"RPM_EMA:{(counter.rpm_ema.y or 0):.1f}",
                    f"LastTTT:{(counter.last_rep_duration or 0):.2f}s",
                    f"LastScore:{counter.last_form_score if counter.last_form_score is not None else '—'}",
                    f"Motivation:{counter.last_motivation if counter.last_motivation is not None else '—'}"
                ])
                print("[STATUS]", status_line)

            # ---- On-screen HUD ----
            panel_h = 30 + 28*len(hud_lines)
            cv2.rectangle(frame, (10, 10), (max(780, w-10), 10+panel_h), (30,30,30), -1)
            for i, line in enumerate(hud_lines):
                y_txt = 40 + 28*i
                cv2.putText(frame, line, (20, y_txt),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Single bar + Motivation
            bar_w = min(500, w-40)
            bar_h = 22
            bx = 20
            by = h - 60
            rep_col  = color_for_value_sec(counter.last_rep_duration, CFG_UI["rep_good"], CFG_UI["rep_bad"])
            draw_hbar(frame, bx, by, bar_w, bar_h,
                      counter.last_rep_duration, CFG_UI["rep_duration_range"],
                      "Rep Duration (Top→Top)", rep_col, invert=True)
            badge_x = bx + bar_w + 10 if bx+bar_w+240 < w else w-240
            draw_motivation_badge(frame, badge_x, by-4, counter.last_motivation)

            # DONE overlay
            if workout.phase == "DONE":
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, int(h*0.35)), (w, int(h*0.65)), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                final_score = workout.final_score() or 0
                final_label = label_from_score(final_score)
                cv2.putText(frame, "WORKOUT COMPLETE!", (int(w*0.15), int(h*0.48)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60,180,60), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Final Quality Score: {final_score}/100 ({final_label})",
                            (int(w*0.12), int(h*0.58)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,215,255), 3, cv2.LINE_AA)

            cv2.imshow("Push-up Rep Counter (realistic scoring + Groq coach)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                counter = PushupCounter(CFG, CFG_SPEED)
                workout = Workout(CFG_WORKOUT["reps_per_set"], CFG_WORKOUT["num_sets"], CFG_WORKOUT["rest_sec"])
                print("[RESET] counters & workout reset.")
            if key == ord('p'):
                print("---- Last 5 reps (raw) ----")
                for r in list(counter.rep_history)[-5:]:
                    print(r)

    finally:
        try:
            pose.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
