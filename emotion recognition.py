# Optimized + cleaned (same logic, way less CPU + no duplicate imshow + correct shutdown)
# - Uses direct landmark indexing (no per-point if/elif jungle)
# - Optional dlib correlation-tracker to avoid full face detection every frame
# - Removes spammy prints (toggle DEBUG)
# - Fixes: stopping the *actual* VideoStream (vs.stop()), not a new one

import os
import time
import cv2
import dlib
import numpy as np
import pygame
import imutils
from imutils.video import VideoStream
from imutils import face_utils

# =========================
# CONFIG
# =========================
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
AUDIO_FILE = "Smile.mp3"

CAM_SRC = 0
RESIZE_WIDTH = 520          # lower = faster (e.g., 400–600)
DETECT_EVERY = 7            # run HOG detector every N frames, track in-between
TRACK_QUALITY_MIN = 6.5     # correlation tracker quality threshold

DRAW_LANDMARKS = True       # set False for speed
DRAW_INDEX = False          # True is slow
DEBUG = False               # prints / verbose overlay

HEAD_STILL_PX = 10

# thresholds (kept close to your original)
SMILE_BIG_JUMP = 15
SMILE_SMOOTH_THR = 6
SMILE_EVENT_MIN = 10
SMILE_EVENT_MAX = 50

EYE_BIG_JUMP = 2
EYE_SMOOTH_THR = 1
EYE_EVENT_MIN = 2.5
EYE_EVENT_MAX = 5.0

BROW_EVENT_THR = 3
BOTH_EYES_SUM_MIN = 2
BOTH_EYES_SUM_MAX = 4
BOTH_EYES_BALANCE_THR = 0.5

# landmark indices (0-based)
MOUTH_L, MOUTH_R = 48, 54          # (49,55) in your 1-based counter
L_EYE_T, L_EYE_B = 37, 40          # (38,41)
R_EYE_T, R_EYE_B = 43, 46          # (44,47)
BROW_PT = 19                        # (20)
ANCHOR_PT = 0                       # (1)

# =========================
# Helpers
# =========================
def l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def pick_largest(rects):
    return max(rects, key=lambda r: r.width() * r.height())

def clamp_rect(rect, w, h):
    l = max(0, rect.left())
    t = max(0, rect.top())
    r = min(w - 1, rect.right())
    b = min(h - 1, rect.bottom())
    if r <= l or b <= t:
        return None
    return dlib.rectangle(l, t, r, b)

def update_baseline(base, curr, diff, big_jump_thr, smooth_thr):
    if base is None or diff > big_jump_thr:
        return curr
    if diff < smooth_thr:
        return 0.5 * (base + curr)
    return base

# =========================
# Init
# =========================
cv2.setUseOptimized(True)

if not os.path.exists(SHAPE_PREDICTOR):
    raise FileNotFoundError(f"Missing predictor file: {SHAPE_PREDICTOR}")

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

pygame.init()
audio_ok = False
try:
    pygame.mixer.init()
    if os.path.exists(AUDIO_FILE):
        pygame.mixer.music.load(AUDIO_FILE)
        audio_ok = True
    else:
        print(f"[WARN] Audio file not found: {AUDIO_FILE} (continuing without sound)")
except Exception as e:
    print(f"[WARN] pygame mixer init/load failed: {e} (continuing without sound)")

print("[INFO] camera sensor warming up...")
vs = VideoStream(src=CAM_SRC).start()
time.sleep(1.5)

# dlib tracker (speed boost)
tracker = dlib.correlation_tracker()
tracking = False
last_rect = None

# state / baselines
base_smile = None
base_leye = None
base_reye = None

prev_anchor = None
prev_brow_rel = None

count_smile = 0
count_eact = 0
count_be = 0

frame_idx = 0
t0 = time.perf_counter()
fps = 0.0
audio_playing = False

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame_idx += 1

        # resize for speed + stable thresholds
        frame = imutils.resize(frame, width=RESIZE_WIDTH)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decide rect: detect every N frames, track otherwise
        rect = None
        if (not tracking) or (frame_idx % DETECT_EVERY == 0):
            rects = detector(gray, 0)
            if len(rects) > 0:
                rect = pick_largest(rects)
                # start tracker on RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tracker.start_track(rgb, rect)
                tracking = True
            else:
                tracking = False
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q = tracker.update(rgb)
            pos = tracker.get_position()
            tr = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
            tr = clamp_rect(tr, w, h)
            if tr is None or q < TRACK_QUALITY_MIN:
                tracking = False
            else:
                rect = tr

        action = None
        e = s = be = le = re = 0

        if rect is not None:
            shape = predictor(gray, rect)
            pts = face_utils.shape_to_np(shape)  # (68,2)

            if DRAW_LANDMARKS:
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
                    if DRAW_INDEX:
                        cv2.putText(frame, str(i + 1), (int(x), int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # anchor (matches your "y-40" trick)
            anchor = (int(pts[ANCHOR_PT][0]), int(pts[ANCHOR_PT][1]) - 40)
            head_dx = head_dy = 0
            if prev_anchor is not None:
                head_dx = abs(anchor[0] - prev_anchor[0])
                head_dy = abs(anchor[1] - prev_anchor[1])
            head_still = (prev_anchor is None) or (head_dx < HEAD_STILL_PX and head_dy < HEAD_STILL_PX)

            # distances
            smile = l2(pts[MOUTH_L], pts[MOUTH_R])
            leye = l2(pts[L_EYE_T], pts[L_EYE_B])
            reye = l2(pts[R_EYE_T], pts[R_EYE_B])

            # diffs vs baselines (computed BEFORE updating baselines)
            diff_smile = 0.0 if base_smile is None else abs(smile - base_smile)
            diff_leye = 0.0 if base_leye is None else abs(leye - base_leye)
            diff_reye = 0.0 if base_reye is None else abs(reye - base_reye)

            # brow movement (your diff_up logic)
            brow_rel = int(pts[BROW_PT][1]) - anchor[1]  # (y20 - y1)
            diff_up = 0
            if prev_brow_rel is not None:
                diff_up = prev_brow_rel - brow_rel  # (y_20 - y20)
            # both-eye condition (kept close)
            diff_eye = 0
            if base_leye is not None and base_reye is not None:
                balance = abs((reye - leye) - (base_reye - base_leye))
                if (diff_leye + diff_reye > BOTH_EYES_SUM_MIN and
                    diff_leye + diff_reye < BOTH_EYES_SUM_MAX and
                    balance < BOTH_EYES_BALANCE_THR):
                    diff_eye = 1

            # decision logic (same priority order)
            if head_still:
                if frame_idx > 1 and (diff_smile > SMILE_EVENT_MIN) and (diff_smile < SMILE_EVENT_MAX):
                    action = "Smile"
                    s = 1
                elif diff_up > BROW_EVENT_THR:
                    action = "eye act"
                    e = 1
                elif diff_eye == 1:
                    action = "BothEye"
                    be = 1
                elif (diff_leye > EYE_EVENT_MIN) and (diff_leye < EYE_EVENT_MAX):
                    action = "Reye"  # kept your original label (even though it's left-eye diff)
                    le = 1
                elif (diff_reye > EYE_EVENT_MIN) and (diff_reye < EYE_EVENT_MAX):
                    action = "Leye"
                    re = 1

            # update baselines (same spirit as your code)
            base_smile = update_baseline(base_smile, smile, diff_smile, SMILE_BIG_JUMP, SMILE_SMOOTH_THR)
            base_leye  = update_baseline(base_leye,  leye,  diff_leye,  EYE_BIG_JUMP,   EYE_SMOOTH_THR)
            base_reye  = update_baseline(base_reye,  reye,  diff_reye,  EYE_BIG_JUMP,   EYE_SMOOTH_THR)

            # update prevs
            prev_anchor = anchor
            prev_brow_rel = brow_rel

            # overlay
            if action is not None:
                cv2.putText(frame, action, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2)
                if action == "Smile":
                    cv2.imshow("selfie1", frame)

            # counters + audio behavior
            if e:
                count_eact += 1
                if audio_ok and (not audio_playing):
                    pygame.mixer.music.play(-1)
                    audio_playing = True
            else:
                # stop audio if it was playing and eye-act is not active
                if audio_ok and audio_playing and action in ("Leye", "Reye"):
                    pygame.mixer.music.stop()
                    audio_playing = False

            if s:
                count_smile += 1
            elif be:
                count_be += 1

            if DEBUG:
                cv2.putText(frame,
                            f"dx={head_dx} dy={head_dy}  dSmile={diff_smile:.1f} dL={diff_leye:.2f} dR={diff_reye:.2f} dUp={diff_up}",
                            (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        else:
            cv2.putText(frame, "No face", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            tracking = False

        # FPS + counters
        t1 = time.perf_counter()
        dt = t1 - t0
        if dt >= 0.5:
            fps = frame_idx / (t1 - (t1 - dt))  # lightweight-ish; not perfect but ok
            t0 = t1
            frame_idx = 0

        cv2.putText(frame, f"Smile:{count_smile}  EyeAct:{count_eact}  BothEye:{count_be}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("l"):
            DRAW_LANDMARKS = not DRAW_LANDMARKS
        elif key == ord("i"):
            DRAW_INDEX = not DRAW_INDEX
        elif key == ord("d"):
            DEBUG = not DEBUG
        elif key == ord("r"):
            base_smile = base_leye = base_reye = None
            prev_anchor = None
            prev_brow_rel = None
            tracking = False

finally:
    # clean shutdown
    try:
        if audio_ok:
            pygame.mixer.music.stop()
    except Exception:
        pass
    pygame.quit()

    try:
        vs.stop()  # IMPORTANT: stop the existing stream
    except Exception:
        pass

    cv2.destroyAllWindows()
