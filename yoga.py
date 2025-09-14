# import cv2, time, math, numpy as np, threading, traceback, sys
# from collections import deque

# # ===================== Easy-mode tuning =====================
# EASE = {
#     "advance_threshold": 0.55,   # lower => easier (try 0.50 if needed)
#     "stability_sec": 0.40,       # shorter => faster transitions
#     "hysteresis_bonus": 0.08,    # keeps progress if you're close
#     "vis_thresh": 0.25           # lower => accept noisier landmarks
# }
# HOLD_SECONDS = 15  # final hold

# # ===================== Geometry helpers =====================
# def angle(a, b, c):
#     a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
#     ba, bc = a - b, c - b
#     nba = ba / (np.linalg.norm(ba) + 1e-9)
#     nbc = bc / (np.linalg.norm(bc) + 1e-9)
#     cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
#     return math.degrees(math.acos(cosang))

# def torso_forward_deg(L):
#     sh = ((np.array(L["left_shoulder"]) + np.array(L["right_shoulder"])) / 2.0)
#     hp = ((np.array(L["left_hip"]) + np.array(L["right_hip"])) / 2.0)
#     v = hp - sh
#     n = np.linalg.norm(v) + 1e-9
#     horiz = abs(v[0]) / n
#     return math.degrees(math.asin(np.clip(horiz, 0, 1)))

# def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

# # ===================== Threaded camera =====================
# class FrameGrabber:
#     def __init__(self, index_order=(0,1,2), width=960, height=540, fps=30):
#         self.index_order = index_order
#         self.width, self.height, self.fps = width, height, fps
#         self.cap = None
#         self.frame = None
#         self.lock = threading.Lock()
#         self.alive = False
#         self.fail = 0

#     def _open(self):
#         for backend in [cv2.CAP_DSHOW, 0]:
#             for idx in self.index_order:
#                 cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
#                 if cap.isOpened():
#                     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
#                     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
#                     cap.set(cv2.CAP_PROP_FPS,          self.fps)
#                     try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#                     except: pass
#                     self.cap = cap
#                     print(f"[CAM] Opened camera {idx} ({'DSHOW' if backend==cv2.CAP_DSHOW else 'DEFAULT'})")
#                     return True
#                 else:
#                     cap.release()
#         return False

#     def start(self):
#         if not self._open():
#             print("[ERR] No camera."); return False
#         self.alive = True
#         threading.Thread(target=self._loop, daemon=True).start()
#         return True

#     def _loop(self):
#         while self.alive:
#             if self.cap is None or not self.cap.isOpened():
#                 time.sleep(0.2)
#                 if not self._open(): continue
#             ok, f = self.cap.read()
#             if not ok:
#                 self.fail += 1
#                 if self.fail >= 10:
#                     try: self.cap.release()
#                     except: pass
#                     self.cap = None
#                     self.fail = 0
#                 time.sleep(0.01); continue
#             self.fail = 0
#             with self.lock:
#                 self.frame = f

#     def get(self):
#         with self.lock:
#             return None if self.frame is None else self.frame.copy()

#     def stop(self):
#         self.alive = False
#         if self.cap is not None:
#             try: self.cap.release()
#             except: pass
#         self.cap = None

# # ===================== Pose utils =====================
# def build_L(res, mp_pose, w, h, vis_thresh=0.25):
#     L = {}
#     lm = res.pose_landmarks.landmark
#     idx = mp_pose.PoseLandmark
#     names = {
#         "left_hip": idx.LEFT_HIP, "right_hip": idx.RIGHT_HIP,
#         "left_knee": idx.LEFT_KNEE, "right_knee": idx.RIGHT_KNEE,
#         "left_ankle": idx.LEFT_ANKLE, "right_ankle": idx.RIGHT_ANKLE,
#         "left_shoulder": idx.LEFT_SHOULDER, "right_shoulder": idx.RIGHT_SHOULDER,
#         "left_elbow": idx.LEFT_ELBOW, "right_elbow": idx.RIGHT_ELBOW,
#         "left_wrist": idx.LEFT_WRIST, "right_wrist": idx.RIGHT_WRIST,
#         "nose": idx.NOSE
#     }
#     for k, vi in names.items():
#         v = lm[vi.value]
#         if v.visibility >= vis_thresh:
#             L[k] = (v.x * w, v.y * h)
#     return L

# def have(L, keys): return all(k in L for k in keys)
# def avg_y(L, a, b): return (L[a][1] + L[b][1]) / 2.0

# # ===================== Step scoring (0..1) =====================
# def score_step0_standing(L):
#     if not have(L, ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_wrist","right_wrist"]):
#         return 0.0, "Get fully in frame."
#     lk = angle(L["left_hip"], L["left_knee"], L["left_ankle"])
#     rk = angle(L["right_hip"], L["right_knee"], L["right_ankle"])
#     hips_y = avg_y(L, "left_hip", "right_hip")
#     wrists_down = (L["left_wrist"][1] > hips_y + 10) + (L["right_wrist"][1] > hips_y + 10)
#     knees = (min(lk, rk) - 140) / 40.0  # 140..180+ -> 0..1
#     wrists = wrists_down / 2.0
#     s = 0.6*knees + 0.4*wrists
#     return clamp01(s), "Arms by side, knees straight."

# def score_step1_bend_and_reach(L):
#     need = ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_wrist","right_wrist","left_shoulder","right_shoulder"]
#     if not have(L, need): return 0.0, "Bend and reach toward ankles."
#     lk = angle(L["left_hip"], L["left_knee"], L["left_ankle"])
#     rk = angle(L["right_hip"], L["right_knee"], L["right_ankle"])
#     bend = 1.0 - (min(lk, rk) - 80)/50.0       # 80..130 -> ~1..0
#     bend = clamp01(bend)
#     ankles_y = avg_y(L, "left_ankle", "right_ankle")
#     wrists_low = ((L["left_wrist"][1] > ankles_y - 50) + (L["right_wrist"][1] > ankles_y - 50)) / 2.0
#     lean = torso_forward_deg(L)
#     lean_score = clamp01((lean - 5)/20.0)      # ≥5°..25° good
#     s = max(bend, wrists_low)*0.8 + 0.2*lean_score
#     return clamp01(s), "Bend knees / reach down."

# def score_step2_sweep_arms_up(L):
#     if not have(L, ["left_elbow","right_elbow","left_wrist","right_wrist","left_shoulder","right_shoulder"]):
#         return 0.0, "Lift arms up."
#     shoulder_y = avg_y(L, "left_shoulder", "right_shoulder")
#     up_w = (L["left_wrist"][1] < shoulder_y - 10) + (L["right_wrist"][1] < shoulder_y - 10)
#     up_e = (L["left_elbow"][1] < shoulder_y - 10) + (L["right_elbow"][1] < shoulder_y - 10)
#     count = max(up_w, up_e)
#     both_bonus = 0.25 if (up_w == 2 and up_e == 2) else 0.0
#     s = clamp01(count/2.0 + both_bonus)
#     return s, "Sweep arms high."

# def score_step3_open_chest(L):
#     need = ["left_wrist","right_wrist","left_elbow","right_elbow","left_shoulder","right_shoulder","left_hip","right_hip"]
#     if not have(L, need): return 0.0, "Arms up, lift chest."
#     shoulder_y = avg_y(L, "left_shoulder", "right_shoulder")
#     up = ((L["left_wrist"][1] < shoulder_y - 10) + (L["right_wrist"][1] < shoulder_y - 10) +
#           (L["left_elbow"][1] < shoulder_y - 10) + (L["right_elbow"][1] < shoulder_y - 10)) / 4.0
#     lean = torso_forward_deg(L)
#     lean_ok = clamp01(1.0 - (lean/25.0)*0.8)   # upright -> higher
#     s = 0.7*up + 0.3*lean_ok
#     return clamp01(s), "Open chest / keep arms high."

# # ===================== Main app =====================
# def main():
#     import mediapipe as mp
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     grab = FrameGrabber()
#     if not grab.start(): return

#     pose = mp_pose.Pose(model_complexity=1,
#                         min_detection_confidence=0.5,
#                         min_tracking_confidence=0.5,
#                         enable_segmentation=False)

#     window = "Yoga — Easy Stepper (ESC to quit)"
#     fps_deq = deque(maxlen=30)

#     STEP_TXT = [
#         "Step 1: Stand tall (arms down).",
#         "Step 2: Bend knees & reach toward the floor.",
#         "Step 3: Sweep arms up.",
#         "Step 4: Open chest (upright, arms high).",
#         "Hold 15s: Maintain the final posture."
#     ]

#     step = 0
#     t_hold_start = 0.0
#     t_stable = 0.0
#     thr = EASE["advance_threshold"]
#     need_stable = EASE["stability_sec"]
#     hysteresis = EASE["hysteresis_bonus"]

#     finished = False
#     final_frame = None   # snapshot to freeze the window

#     print("[INFO] Easy stepper started. Finish will freeze the window until ESC.")
#     try:
#         while True:
#             # If finished, keep showing the frozen final_frame until ESC
#             if finished and final_frame is not None:
#                 cv2.imshow(window, final_frame)
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == 27:
#                     print("[EXIT] ESC pressed."); break
#                 try:
#                     if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
#                         print("[EXIT] Window closed."); break
#                 except:
#                     pass
#                 continue

#             t0 = time.time()
#             frame = grab.get()
#             if frame is None:
#                 time.sleep(0.01); continue

#             h, w = frame.shape[:2]
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb)

#             L = {}
#             if res.pose_landmarks:
#                 L = build_L(res, mp_pose, w, h, vis_thresh=EASE["vis_thresh"])

#             s0, tip0 = score_step0_standing(L) if L else (0.0, "Get in frame.")
#             s1, tip1 = score_step1_bend_and_reach(L) if L else (0.0, "Bend & reach.")
#             s2, tip2 = score_step2_sweep_arms_up(L) if L else (0.0, "Arms up.")
#             s3, tip3 = score_step3_open_chest(L) if L else (0.0, "Open chest.")

#             if res.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
#                     connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
#                 )

#             # Active step progression
#             step_scores = [s0, s1, s2, s3]
#             if step < 4:
#                 score = step_scores[step]
#                 if t_stable > 0: score = clamp01(score + hysteresis)
#                 now = time.time()
#                 if score >= thr:
#                     if t_stable == 0.0: t_stable = now
#                     if now - t_stable >= need_stable:
#                         step += 1
#                         t_stable = 0.0
#                         if step == 4:
#                             t_hold_start = time.time()
#                         print(f"✓ Advance → {STEP_TXT[step]}")
#                 else:
#                     t_stable = 0.0

#             # Final 15s hold
#             hold_left = 0
#             if step == 4:
#                 if s3 >= (thr - 0.10):
#                     elapsed = time.time() - t_hold_start
#                     hold_left = max(0, HOLD_SECONDS - int(elapsed))
#                     # === SUCCESS PATH: freeze window ===
#                     if elapsed >= HOLD_SECONDS:
#                         # Stamp the success message, snapshot the frame, stop camera, then freeze.
#                         cv2.putText(frame, "YOGA FINISHED ✅", (int(w*0.18), int(h*0.20)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4, cv2.LINE_AA)
#                         cv2.putText(frame, "Great job holding 15 seconds!", (int(w*0.18), int(h*0.28)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
#                         final_frame = frame.copy()
#                         grab.stop()           # release camera
#                         finished = True       # enter freeze mode
#                         # Skip drawing more UI; show frozen frame in loop at top.
#                         continue
#                 else:
#                     t_hold_start = time.time()
#                     hold_left = HOLD_SECONDS

#             # UI text
#             top = STEP_TXT[step]
#             cv2.putText(frame, top, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#             # Tips
#             if step == 0:
#                 sub = tip0 if s0 < thr else "Nice. Hold briefly…"
#             elif step == 1:
#                 sub = tip1 if s1 < thr else "Good bend/reach. Hold…"
#             elif step == 2:
#                 sub = tip2 if s2 < thr else "Arms high. Hold…"
#             elif step == 3:
#                 sub = tip3 if s3 < thr else "Open chest. Hold…"
#             else:
#                 sub = f"Hold the posture — {hold_left:02d}s" if s3 >= (thr - 0.10) else "Regain posture to resume timer."
#             color = (0,200,0) if step==4 and s3 >= (thr-0.10) and hold_left<=5 else (0,215,255)
#             cv2.putText(frame, sub, (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#             # Left progress bar
#             bar_x, bar_y, bar_w, bar_h = 20, 110, 20, int(h*0.72)
#             cv2.rectangle(frame, (bar_x-2,bar_y-2), (bar_x+bar_w+2,bar_y+bar_h+2), (70,70,70), 2)
#             filled = int(bar_h * ((step + (0 if step<4 else 1))/5.0))
#             cv2.rectangle(frame, (bar_x, bar_y+bar_h-filled), (bar_x+bar_w, bar_y+bar_h), (0,180,0), -1)

#             # Current step score meter (right bottom)
#             if step < 4:
#                 score = step_scores[step]
#                 meter_w = 220; meter_h = 18; mx = w - meter_w - 30; my = h - 30
#                 cv2.rectangle(frame, (mx, my-meter_h), (mx+meter_w, my), (80,80,80), 2)
#                 fill = int(meter_w * clamp01(score))
#                 cv2.rectangle(frame, (mx, my-meter_h), (mx+fill, my), (0,180,0), -1)
#                 cv2.putText(frame, f"Score {score:.2f} / Thr {thr:.2f}", (mx, my-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

#             # Hold ring (top-right) in final step
#             if step == 4:
#                 center = (w-90, 90); radius = 50
#                 cv2.circle(frame, center, radius, (160,160,160), 2)
#                 elapsed = time.time() - t_hold_start
#                 pct = 0.0 if s3 < (thr-0.10) else min(1.0, max(0.0, elapsed / HOLD_SECONDS))
#                 theta = int(360 * pct)
#                 cv2.ellipse(frame, center, (radius, radius), -90, 0, theta, (0,200,0), 6)
#                 txt = f"{max(0,HOLD_SECONDS-int(elapsed))}s" if s3 >= (thr-0.10) else f"{HOLD_SECONDS}s"
#                 cv2.putText(frame, txt, (center[0]-22, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

#             # FPS
#             fps = 1.0 / max(1e-3, time.time()-t0)
#             fps_deq.append(fps)
#             fps_s = sum(fps_deq)/len(fps_deq)
#             cv2.putText(frame, f"{fps_s:.1f} FPS", (w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

#             cv2.imshow(window, frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27:
#                 print("[EXIT] ESC pressed."); break
#             try:
#                 if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
#                     print("[EXIT] Window closed."); break
#             except:
#                 pass

#         grab.stop()
#         cv2.destroyAllWindows()

#     except Exception:
#         print("[FATAL] Unhandled exception.")
#         traceback.print_exc()
#         try: cv2.destroyAllWindows()
#         except: pass
#         grab.stop()
#         sys.exit(1)

# if __name__ == "__main__":
#     print("[BOOT] Yoga Easy Stepper — Stand → Bend/Reach → Arms Up → Open Chest → Hold 15s")
#     print("Finish behavior: shows 'YOGA FINISHED ✅' and freezes the window until you press ESC.")
#     main()

import cv2, time, math, numpy as np, threading, traceback, sys
from collections import deque

# ===================== Easy-mode tuning =====================
EASE = {
    "advance_threshold": 0.55,   # lower => easier (try 0.50 if needed)
    "stability_sec": 0.40,       # shorter => faster transitions
    "hysteresis_bonus": 0.08,    # keeps progress if you're close
    "vis_thresh": 0.25           # lower => accept noisier landmarks
}
HOLD_SECONDS = 15  # final hold

# ===================== Geometry helpers =====================
def angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def torso_forward_deg(L):
    sh = ((np.array(L["left_shoulder"]) + np.array(L["right_shoulder"])) / 2.0)
    hp = ((np.array(L["left_hip"]) + np.array(L["right_hip"])) / 2.0)
    v = hp - sh
    n = np.linalg.norm(v) + 1e-9
    horiz = abs(v[0]) / n
    return math.degrees(math.asin(np.clip(horiz, 0, 1)))

def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

# ===================== Threaded camera =====================
class FrameGrabber:
    def __init__(self, index_order=(0,1,2), width=960, height=540, fps=30):
        self.index_order = index_order
        self.width, self.height, self.fps = width, height, fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.alive = False
        self.fail = 0

    def _open(self):
        for backend in [cv2.CAP_DSHOW, 0]:
            for idx in self.index_order:
                cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS,          self.fps)
                    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except: pass
                    self.cap = cap
                    print(f"[CAM] Opened camera {idx} ({'DSHOW' if backend==cv2.CAP_DSHOW else 'DEFAULT'})")
                    return True
                else:
                    cap.release()
        return False

    def start(self):
        if not self._open():
            print("[ERR] No camera."); return False
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        while self.alive:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                if not self._open(): continue
            ok, f = self.cap.read()
            if not ok:
                self.fail += 1
                if self.fail >= 10:
                    try: self.cap.release()
                    except: pass
                    self.cap = None
                    self.fail = 0
                time.sleep(0.01); continue
            self.fail = 0
            with self.lock:
                self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.alive = False
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = None

# ===================== Pose utils =====================
def build_L(res, mp_pose, w, h, vis_thresh=0.25):
    L = {}
    lm = res.pose_landmarks.landmark
    idx = mp_pose.PoseLandmark
    names = {
        "left_hip": idx.LEFT_HIP, "right_hip": idx.RIGHT_HIP,
        "left_knee": idx.LEFT_KNEE, "right_knee": idx.RIGHT_KNEE,
        "left_ankle": idx.LEFT_ANKLE, "right_ankle": idx.RIGHT_ANKLE,
        "left_shoulder": idx.LEFT_SHOULDER, "right_shoulder": idx.RIGHT_SHOULDER,
        "left_elbow": idx.LEFT_ELBOW, "right_elbow": idx.RIGHT_ELBOW,
        "left_wrist": idx.LEFT_WRIST, "right_wrist": idx.RIGHT_WRIST,
        "nose": idx.NOSE
    }
    for k, vi in names.items():
        v = lm[vi.value]
        if v.visibility >= vis_thresh:
            L[k] = (v.x * w, v.y * h)
    return L

def have(L, keys): return all(k in L for k in keys)
def avg_y(L, a, b): return (L[a][1] + L[b][1]) / 2.0

# ===================== Step scoring (0..1) =====================
def score_step0_standing(L):
    if not have(L, ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_wrist","right_wrist"]):
        return 0.0, "Get fully in frame."
    lk = angle(L["left_hip"], L["left_knee"], L["left_ankle"])
    rk = angle(L["right_hip"], L["right_knee"], L["right_ankle"])
    hips_y = avg_y(L, "left_hip", "right_hip")
    wrists_down = (L["left_wrist"][1] > hips_y + 10) + (L["right_wrist"][1] > hips_y + 10)
    knees = (min(lk, rk) - 140) / 40.0  # 140..180+ -> 0..1
    wrists = wrists_down / 2.0
    s = 0.6*knees + 0.4*wrists
    return clamp01(s), "Arms by side, knees straight."

def score_step1_bend_and_reach(L):
    need = ["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_wrist","right_wrist","left_shoulder","right_shoulder"]
    if not have(L, need): return 0.0, "Bend and reach toward ankles."
    lk = angle(L["left_hip"], L["left_knee"], L["left_ankle"])
    rk = angle(L["right_hip"], L["right_knee"], L["right_ankle"])
    bend = 1.0 - (min(lk, rk) - 80)/50.0       # 80..130 -> ~1..0
    bend = clamp01(bend)
    ankles_y = avg_y(L, "left_ankle", "right_ankle")
    wrists_low = ((L["left_wrist"][1] > ankles_y - 50) + (L["right_wrist"][1] > ankles_y - 50)) / 2.0
    lean = torso_forward_deg(L)
    lean_score = clamp01((lean - 5)/20.0)      # ≥5°..25° good
    s = max(bend, wrists_low)*0.8 + 0.2*lean_score
    return clamp01(s), "Bend knees / reach down."

def score_step2_sweep_arms_up(L):
    if not have(L, ["left_elbow","right_elbow","left_wrist","right_wrist","left_shoulder","right_shoulder"]):
        return 0.0, "Lift arms up."
    shoulder_y = avg_y(L, "left_shoulder", "right_shoulder")
    up_w = (L["left_wrist"][1] < shoulder_y - 10) + (L["right_wrist"][1] < shoulder_y - 10)
    up_e = (L["left_elbow"][1] < shoulder_y - 10) + (L["right_elbow"][1] < shoulder_y - 10)
    count = max(up_w, up_e)
    both_bonus = 0.25 if (up_w == 2 and up_e == 2) else 0.0
    s = clamp01(count/2.0 + both_bonus)
    return s, "Sweep arms high."

def score_step3_open_chest(L):
    need = ["left_wrist","right_wrist","left_elbow","right_elbow","left_shoulder","right_shoulder","left_hip","right_hip"]
    if not have(L, need): return 0.0, "Arms up, lift chest."
    shoulder_y = avg_y(L, "left_shoulder", "right_shoulder")
    up = ((L["left_wrist"][1] < shoulder_y - 10) + (L["right_wrist"][1] < shoulder_y - 10) +
          (L["left_elbow"][1] < shoulder_y - 10) + (L["right_elbow"][1] < shoulder_y - 10)) / 4.0
    lean = torso_forward_deg(L)
    lean_ok = clamp01(1.0 - (lean/25.0)*0.8)   # upright -> higher
    s = 0.7*up + 0.3*lean_ok
    return clamp01(s), "Open chest / keep arms high."

# ===================== Main app =====================
def main():
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    grab = FrameGrabber()
    if not grab.start(): return

    pose = mp_pose.Pose(model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        enable_segmentation=False)

    window = "Yoga — Easy Stepper (ESC to quit)"
    fps_deq = deque(maxlen=30)

    STEP_TXT = [
        "Step 1: Stand tall (arms down).",
        "Step 2: Bend knees & reach toward the floor.",
        "Step 3: Sweep arms up.",
        "Step 4: Open chest (upright, arms high).",
        "Hold 15s: Maintain the final posture."
    ]

    step = 0
    t_hold_start = 0.0
    t_stable = 0.0
    thr = EASE["advance_threshold"]
    need_stable = EASE["stability_sec"]
    hysteresis = EASE["hysteresis_bonus"]

    finished = False
    final_frame = None   # snapshot to freeze the window

    print("[INFO] Easy stepper started. Finish will freeze the window until ESC.")
    try:
        while True:
            # If finished, keep showing the frozen final_frame until ESC
            if finished and final_frame is not None:
                cv2.imshow(window, final_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[EXIT] ESC pressed."); break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        print("[EXIT] Window closed."); break
                except:
                    pass
                continue

            t0 = time.time()
            frame = grab.get()
            if frame is None:
                time.sleep(0.01); continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            L = {}
            if res.pose_landmarks:
                L = build_L(res, mp_pose, w, h, vis_thresh=EASE["vis_thresh"])

            s0, tip0 = score_step0_standing(L) if L else (0.0, "Get in frame.")
            s1, tip1 = score_step1_bend_and_reach(L) if L else (0.0, "Bend & reach.")
            s2, tip2 = score_step2_sweep_arms_up(L) if L else (0.0, "Arms up.")
            s3, tip3 = score_step3_open_chest(L) if L else (0.0, "Open chest.")

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
                )

            # Active step progression
            step_scores = [s0, s1, s2, s3]
            if step < 4:
                score = step_scores[step]
                if t_stable > 0: score = clamp01(score + hysteresis)
                now = time.time()
                if score >= thr:
                    if t_stable == 0.0: t_stable = now
                    if now - t_stable >= need_stable:
                        step += 1
                        t_stable = 0.0
                        if step == 4:
                            t_hold_start = time.time()
                        print(f"✓ Advance → {STEP_TXT[step]}")
                else:
                    t_stable = 0.0

            # Final 15s hold
            hold_left = 0
            if step == 4:
                if s3 >= (thr - 0.10):
                    elapsed = time.time() - t_hold_start
                    hold_left = max(0, HOLD_SECONDS - int(elapsed))
                    # === SUCCESS PATH: freeze window ===
                    if elapsed >= HOLD_SECONDS:
                        # Stamp the success message, snapshot the frame, stop camera, then freeze.
                        cv2.putText(frame, "YOGA FINISHED ✅", (int(w*0.18), int(h*0.20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(frame, "Great job holding 15 seconds!", (int(w*0.18), int(h*0.28)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                        final_frame = frame.copy()
                        grab.stop()           # release camera
                        finished = True       # enter freeze mode
                        # Skip drawing more UI; show frozen frame in loop at top.
                        continue
                else:
                    t_hold_start = time.time()
                    hold_left = HOLD_SECONDS

            # UI text
            top = STEP_TXT[step]
            cv2.putText(frame, top, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Tips
            if step == 0:
                sub = tip0 if s0 < thr else "Nice. Hold briefly…"
            elif step == 1:
                sub = tip1 if s1 < thr else "Good bend/reach. Hold…"
            elif step == 2:
                sub = tip2 if s2 < thr else "Arms high. Hold…"
            elif step == 3:
                sub = tip3 if s3 < thr else "Open chest. Hold…"
            else:
                sub = f"Hold the posture — {hold_left:02d}s" if s3 >= (thr - 0.10) else "Regain posture to resume timer."
            color = (0,200,0) if step==4 and s3 >= (thr-0.10) and hold_left<=5 else (0,215,255)
            cv2.putText(frame, sub, (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Left progress bar
            bar_x, bar_y, bar_w, bar_h = 20, 110, 20, int(h*0.72)
            cv2.rectangle(frame, (bar_x-2,bar_y-2), (bar_x+bar_w+2,bar_y+bar_h+2), (70,70,70), 2)
            filled = int(bar_h * ((step + (0 if step<4 else 1))/5.0))
            cv2.rectangle(frame, (bar_x, bar_y+bar_h-filled), (bar_x+bar_w, bar_y+bar_h), (0,180,0), -1)

            # Current step score meter (right bottom)
            if step < 4:
                score = step_scores[step]
                meter_w = 220; meter_h = 18; mx = w - meter_w - 30; my = h - 30
                cv2.rectangle(frame, (mx, my-meter_h), (mx+meter_w, my), (80,80,80), 2)
                fill = int(meter_w * clamp01(score))
                cv2.rectangle(frame, (mx, my-meter_h), (mx+fill, my), (0,180,0), -1)
                cv2.putText(frame, f"Score {score:.2f} / Thr {thr:.2f}", (mx, my-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # Hold ring (top-right) in final step
            if step == 4:
                center = (w-90, 90); radius = 50
                cv2.circle(frame, center, radius, (160,160,160), 2)
                elapsed = time.time() - t_hold_start
                pct = 0.0 if s3 < (thr-0.10) else min(1.0, max(0.0, elapsed / HOLD_SECONDS))
                theta = int(360 * pct)
                cv2.ellipse(frame, center, (radius, radius), -90, 0, theta, (0,200,0), 6)
                txt = f"{max(0,HOLD_SECONDS-int(elapsed))}s" if s3 >= (thr-0.10) else f"{HOLD_SECONDS}s"
                cv2.putText(frame, txt, (center[0]-22, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # FPS
            fps = 1.0 / max(1e-3, time.time()-t0)
            fps_deq.append(fps)
            fps_s = sum(fps_deq)/len(fps_deq)
            cv2.putText(frame, f"{fps_s:.1f} FPS", (w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[EXIT] ESC pressed."); break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    print("[EXIT] Window closed."); break
            except:
                pass

        grab.stop()
        cv2.destroyAllWindows()

    except Exception:
        print("[FATAL] Unhandled exception.")
        traceback.print_exc()
        try: cv2.destroyAllWindows()
        except: pass
        grab.stop()
        sys.exit(1)

if __name__ == "__main__":
    print("[BOOT] Yoga Easy Stepper — Stand → Bend/Reach → Arms Up → Open Chest → Hold 15s")
    print("Finish behavior: shows 'YOGA FINISHED ✅' and freezes the window until you press ESC.")
    main()
