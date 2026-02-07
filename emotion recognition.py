import base64
import bz2
import os
import time
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import av
import cv2
import dlib
import imutils
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from imutils import face_utils
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================
# Model download (auto)
# =========================
MODEL_URLS = [
    "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
]

CACHE_DIR = Path.home() / ".cache" / "dlib_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DAT = CACHE_DIR / "shape_predictor_68_face_landmarks.dat"
PRED_BZ2 = CACHE_DIR / "shape_predictor_68_face_landmarks.dat.bz2"


def _download_file(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _decompress_bz2(src_bz2: Path, dst_dat: Path) -> None:
    with bz2.open(src_bz2, "rb") as f_in, open(dst_dat, "wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)


@st.cache_resource(show_spinner=False)
def ensure_predictor() -> str:
    if PRED_DAT.exists():
        return str(PRED_DAT)

    # download bz2
    last_err = None
    for url in MODEL_URLS:
        try:
            _download_file(url, PRED_BZ2)
            _decompress_bz2(PRED_BZ2, PRED_DAT)
            try:
                PRED_BZ2.unlink(missing_ok=True)
            except Exception:
                pass
            return str(PRED_DAT)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to download predictor. Last error: {last_err}")


@st.cache_resource(show_spinner=False)
def load_models(predictor_path: str):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


# =========================
# Detection config
# =========================
# landmark indices (0-based)
MOUTH_L, MOUTH_R = 48, 54          # (49,55) in your 1-based counter
L_EYE_T, L_EYE_B = 37, 40          # (38,41)
R_EYE_T, R_EYE_B = 43, 46          # (44,47)
BROW_PT = 19                        # (20)
ANCHOR_PT = 0                       # (1)


def l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def clamp_rect(rect: dlib.rectangle, w: int, h: int) -> Optional[dlib.rectangle]:
    l = max(0, rect.left())
    t = max(0, rect.top())
    r = min(w - 1, rect.right())
    b = min(h - 1, rect.bottom())
    if r <= l or b <= t:
        return None
    return dlib.rectangle(l, t, r, b)


def update_baseline(base: Optional[float], curr: float, diff: float, big_jump_thr: float, smooth_thr: float) -> float:
    if base is None or diff > big_jump_thr:
        return curr
    if diff < smooth_thr:
        return 0.5 * (base + curr)
    return base


@dataclass
class Cfg:
    resize_width: int = 520
    detect_every: int = 7
    track_quality_min: float = 6.5

    draw_landmarks: bool = True
    draw_index: bool = False
    draw_bbox: bool = True
    debug: bool = False

    head_still_px: int = 10
    anchor_y_offset: int = 40

    smile_big_jump: float = 15
    smile_smooth: float = 6
    smile_event_min: float = 10
    smile_event_max: float = 50

    eye_big_jump: float = 2
    eye_smooth: float = 1
    eye_event_min: float = 2.5
    eye_event_max: float = 5.0

    brow_event_thr: float = 3
    both_sum_min: float = 2
    both_sum_max: float = 4
    both_balance_thr: float = 0.5

    event_cooldown_s: float = 0.8


# =========================
# Video Processor
# =========================
class GestureProcessor(VideoProcessorBase):
    def __init__(self, detector, predictor, cfg: Cfg):
        self.detector = detector
        self.predictor = predictor
        self.cfg = cfg

        self.tracker = dlib.correlation_tracker()
        self.tracking = False
        self.frame_i = 0

        self.base_smile: Optional[float] = None
        self.base_leye: Optional[float] = None
        self.base_reye: Optional[float] = None

        self.prev_anchor: Optional[Tuple[int, int]] = None
        self.prev_brow_rel: Optional[int] = None

        self._last_emit_ts = 0.0

        self._lock = threading.Lock()
        self._event_id = 0
        self._last_event: Optional[str] = None
        self._counts = {"smile": 0, "eyeact": 0, "both": 0, "leye": 0, "reye": 0}

    def _emit(self, event: str):
        now = time.time()
        if (now - self._last_emit_ts) < self.cfg.event_cooldown_s:
            return
        self._last_emit_ts = now

        with self._lock:
            self._last_event = event
            self._event_id += 1
            if event == "Smile":
                self._counts["smile"] += 1
            elif event == "eye act":
                self._counts["eyeact"] += 1
            elif event == "BothEye":
                self._counts["both"] += 1
            elif event == "Leye":
                self._counts["leye"] += 1
            elif event == "Reye":
                self._counts["reye"] += 1

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "event_id": self._event_id,
                "last_event": self._last_event,
                "counts": dict(self._counts),
            }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # resize for speed
        if self.cfg.resize_width and img.shape[1] > self.cfg.resize_width:
            img = imutils.resize(img, width=self.cfg.resize_width)

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_i += 1

        rect = None

        # detect every N frames; track in-between
        if (not self.tracking) or (self.frame_i % self.cfg.detect_every == 0):
            rects = self.detector(gray, 0)
            if rects:
                rect = max(rects, key=lambda r: r.width() * r.height())
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.tracker.start_track(rgb, rect)
                self.tracking = True
            else:
                self.tracking = False
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q = self.tracker.update(rgb)
            pos = self.tracker.get_position()
            tr = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
            tr = clamp_rect(tr, w, h)
            if tr is None or q < self.cfg.track_quality_min:
                self.tracking = False
            else:
                rect = tr

        if rect is None:
            cv2.putText(img, "No face", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.cfg.draw_bbox:
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        shape = self.predictor(gray, rect)
        pts = face_utils.shape_to_np(shape)

        if self.cfg.draw_landmarks:
            for i, (x, y) in enumerate(pts):
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
                if self.cfg.draw_index:
                    cv2.putText(img, str(i + 1), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # anchor (your y-40 trick)
        anchor_base = pts[ANCHOR_PT]
        anchor = (int(anchor_base[0]), int(anchor_base[1]) - self.cfg.anchor_y_offset)

        head_dx = head_dy = 0
        if self.prev_anchor is not None:
            head_dx = abs(anchor[0] - self.prev_anchor[0])
            head_dy = abs(anchor[1] - self.prev_anchor[1])
        head_still = (self.prev_anchor is None) or (head_dx < self.cfg.head_still_px and head_dy < self.cfg.head_still_px)

        # distances
        smile = l2(pts[MOUTH_L], pts[MOUTH_R])
        leye = l2(pts[L_EYE_T], pts[L_EYE_B])
        reye = l2(pts[R_EYE_T], pts[R_EYE_B])

        diff_smile = 0.0 if self.base_smile is None else abs(smile - self.base_smile)
        diff_leye = 0.0 if self.base_leye is None else abs(leye - self.base_leye)
        diff_reye = 0.0 if self.base_reye is None else abs(reye - self.base_reye)

        # brow movement (your diff_up logic)
        brow_rel = int(pts[BROW_PT][1]) - anchor[1]
        diff_up = 0
        if self.prev_brow_rel is not None:
            diff_up = self.prev_brow_rel - brow_rel

        # both-eye check (kept close to your logic)
        diff_eye = 0
        if self.base_leye is not None and self.base_reye is not None:
            balance = abs((reye - leye) - (self.base_reye - self.base_leye))
            if (diff_leye + diff_reye > self.cfg.both_sum_min and
                diff_leye + diff_reye < self.cfg.both_sum_max and
                balance < self.cfg.both_balance_thr):
                diff_eye = 1

        event = None
        if head_still:
            if self.frame_i > 1 and (diff_smile > self.cfg.smile_event_min) and (diff_smile < self.cfg.smile_event_max):
                event = "Smile"
            elif diff_up > self.cfg.brow_event_thr:
                event = "eye act"
            elif diff_eye == 1:
                event = "BothEye"
            elif (diff_leye > self.cfg.eye_event_min) and (diff_leye < self.cfg.eye_event_max):
                event = "Reye"
            elif (diff_reye > self.cfg.eye_event_min) and (diff_reye < self.cfg.eye_event_max):
                event = "Leye"

        if event:
            self._emit(event)
            cv2.putText(img, event, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2)

        # update baselines
        self.base_smile = update_baseline(self.base_smile, smile, diff_smile, self.cfg.smile_big_jump, self.cfg.smile_smooth)
        self.base_leye = update_baseline(self.base_leye, leye, diff_leye, self.cfg.eye_big_jump, self.cfg.eye_smooth)
        self.base_reye = update_baseline(self.base_reye, reye, diff_reye, self.cfg.eye_big_jump, self.cfg.eye_smooth)

        self.prev_anchor = anchor
        self.prev_brow_rel = brow_rel

        if self.cfg.debug:
            cv2.putText(
                img,
                f"dx={head_dx} dy={head_dy} dSmile={diff_smile:.1f} dL={diff_leye:.2f} dR={diff_reye:.2f} dUp={diff_up}",
                (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# UI
# =========================
st.set_page_config(page_title="Face Gesture Detector", layout="wide")
st.title("Face Gesture Detector (Streamlit Cloud / WebRTC)")

with st.sidebar:
    st.subheader("Performance")
    resize_width = st.slider("Resize width", 320, 960, 520, 20)
    detect_every = st.slider("Detect every N frames (tracking in-between)", 2, 20, 7, 1)

    st.subheader("Overlays")
    draw_landmarks = st.checkbox("Draw landmarks", True)
    draw_index = st.checkbox("Draw landmark indices (slow)", False)
    draw_bbox = st.checkbox("Draw face box", True)
    debug = st.checkbox("Debug text", False)

    st.subheader("Events")
    play_sound = st.checkbox("Play sound on event (browser)", False)
    sound_file = st.text_input("Sound file (in repo)", "Smile.mp3")

cfg = Cfg(
    resize_width=resize_width,
    detect_every=detect_every,
    draw_landmarks=draw_landmarks,
    draw_index=draw_index,
    draw_bbox=draw_bbox,
    debug=debug,
)

# predictor + models
with st.spinner("Preparing dlib model (first run may download ~60MB + decompress)..."):
    predictor_path = ensure_predictor()
detector, predictor = load_models(predictor_path)

audio_b64 = None
if play_sound:
    p = Path(sound_file)
    if p.exists():
        audio_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    else:
        st.sidebar.warning(f"Sound file not found: {sound_file}")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    ctx = webrtc_streamer(
        key="cam",
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: GestureProcessor(detector, predictor, cfg),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Live Events")
    status_ph = st.empty()
    counts_ph = st.empty()
    sound_ph = st.empty()

    seen_id = -1

    # Live update loop (only while playing)
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp is None:
            time.sleep(0.1)
            continue

        state = vp.get_state()
        counts_ph.json(state["counts"])

        eid = state["event_id"]
        ev = state["last_event"]

        if eid != seen_id and ev:
            seen_id = eid
            status_ph.success(f"Detected: {ev}")

            # Try to autoplay sound (browser may block autoplay depending on settings)
            if play_sound and audio_b64:
                sound_ph.empty()
                components.html(
                    f"""
                    <audio autoplay>
                      <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
                    </audio>
                    """,
                    height=0,
                )

        time.sleep(0.15)
