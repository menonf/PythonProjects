"""
photo_classifier.py — Smart photo quality classifier (v2)

Features:
  - FFT-based sharpness + motion-blur detection
  - Emotion scoring from DeepFace analysis
  - Noise estimation (median Laplacian)
  - Colour-cast & local-contrast scoring
  - Backlit-subject detection
  - Red-eye detection
  - EXIF metadata analysis (ISO, shutter speed, flash, auto-rotate)
  - Best-of-duplicate-cluster selection (keeps sharpest)
  - Bad-photo sub-folders classified by rejection reason
  - Configurable profiles (portrait / group / landscape)
  - tqdm progress bars (optional)
  - SQLite result caching
  - HTML gallery report with filters
  - HEIC/HEIF support (requires pillow-heif)

Usage:
    python photos.py [--input DIR] [--min-score N] [--dry-run] [--workers N]
                     [--report] [--html-report] [--profile portrait|group|landscape]
                     [--device cpu|cuda] [--no-cache]
"""

import os
import json
import shutil
import logging
import argparse
import sqlite3
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageOps, ExifTags
import imagehash

# Optional dependencies -------------------------------------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import pillow_heif  # noqa: F401
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False


# ===========================================================================
# Scoring profiles (each set of weights sums to 100)
# ===========================================================================

PROFILES: dict[str, dict] = {
    "portrait": dict(
        weight_sharpness=20, weight_brightness=10, weight_face=25,
        weight_eyes=15, weight_composition=10, weight_emotion=10,
        weight_noise=5, weight_color=5,
        min_faces=1, max_faces=100,
    ),
    "group": dict(
        weight_sharpness=20, weight_brightness=10, weight_face=15,
        weight_eyes=10, weight_composition=15, weight_emotion=10,
        weight_noise=10, weight_color=10,
        min_faces=2, max_faces=50,
    ),
    "landscape": dict(
        weight_sharpness=25, weight_brightness=15, weight_face=0,
        weight_eyes=0, weight_composition=20, weight_emotion=0,
        weight_noise=15, weight_color=25,
        min_faces=0, max_faces=999,
    ),
}


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    input_folder: str = "input_photos"
    good_folder: str = "good_photos"
    bad_folder: str = "bad_photos"
    report_file: str = "report.json"
    html_report_file: str = "report.html"
    profile: str = "portrait"

    # Scoring weights (must sum to 100) — overridden by apply_profile()
    weight_sharpness: int = 20
    weight_brightness: int = 10
    weight_face: int = 25
    weight_eyes: int = 15
    weight_composition: int = 10
    weight_emotion: int = 10
    weight_noise: int = 5
    weight_color: int = 5

    # Quality thresholds
    min_score: int = 60
    min_sharpness: float = 100.0
    min_brightness: float = 50.0
    max_brightness: float = 220.0
    min_resolution: tuple = (640, 480)
    min_faces: int = 1
    max_faces: int = 10
    duplicate_threshold: int = 8

    # Parallelism & device
    workers: int = 4
    device: str = "cpu"

    # Cache
    cache_db: str = ".photo_cache.db"
    enable_cache: bool = True

    # Extensions (HEIC/HEIF if pillow-heif installed)
    extensions: tuple = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".heic", ".heif",
    )

    def apply_profile(self):
        """Override weights & face limits from the named profile."""
        if self.profile in PROFILES:
            for key, val in PROFILES[self.profile].items():
                setattr(self, key, val)


# ===========================================================================
# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Serialise DeepFace calls (TensorFlow is not always thread-safe)
_deepface_lock = threading.Lock()


# ===========================================================================
# Image I/O helpers — read once, reuse everywhere
# ===========================================================================

def load_image(image_path: str):
    """
    Open via PIL (auto-rotate from EXIF), convert to OpenCV BGR + greyscale.
    Returns ``(pil_img, img_bgr, gray)`` or raises on failure.
    """
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img) or pil_img
    rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return pil_img, img_bgr, gray


def read_exif(pil_img: Image.Image) -> dict:
    """Extract useful EXIF metadata."""
    exif: dict = {}
    try:
        raw = pil_img._getexif()
        if raw is None:
            return exif
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        exif["iso"] = tags.get("ISOSpeedRatings")
        exif["aperture"] = tags.get("FNumber")
        exif["shutter_speed"] = tags.get("ExposureTime")
        exif["flash"] = tags.get("Flash")
        exif["datetime"] = tags.get("DateTimeOriginal") or tags.get("DateTime")
        exif["orientation"] = tags.get("Orientation")
    except Exception:
        pass
    return exif


# ===========================================================================
# Individual quality checks  (each returns a 0.0–1.0 score + note string)
# ===========================================================================

# ── Sharpness (FFT + Laplacian + motion-blur) ────────────────────────────

def _fft_high_freq_ratio(gray: np.ndarray) -> float:
    """Fraction of energy outside the low-frequency centre of the spectrum."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    rows, cols = gray.shape
    r = int(min(rows, cols) * 0.05)
    crow, ccol = rows // 2, cols // 2
    low = mag[max(0, crow - r):crow + r, max(0, ccol - r):ccol + r].sum()
    total = mag.sum()
    return float(1.0 - low / total) if total > 0 else 0.0


def _motion_blur_score(gray: np.ndarray) -> tuple[float, str]:
    """Directional gradient imbalance → 1.0 = balanced, 0.0 = heavy blur."""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ex, ey = float(np.mean(sx ** 2)), float(np.mean(sy ** 2))
    if max(ex, ey) == 0:
        return 0.0, "motion=N/A"
    ratio = min(ex, ey) / max(ex, ey)
    direction = "horizontal" if ex < ey else "vertical"
    note = f"motion={ratio:.2f}" + (f"({direction})" if ratio < 0.5 else "")
    return ratio, note


def score_sharpness(gray: np.ndarray) -> tuple[float, str]:
    """Combined sharpness: Laplacian variance + FFT + motion-blur."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_s = min(lap_var / 300.0, 1.0)

    fft_ratio = _fft_high_freq_ratio(gray)
    fft_s = min(fft_ratio / 0.4, 1.0)

    mot_s, mot_n = _motion_blur_score(gray)

    score = 0.4 * lap_s + 0.3 * fft_s + 0.3 * mot_s
    note = f"sharp(lap={lap_var:.0f} fft={fft_ratio:.2f} {mot_n})"
    return score, note


# ── Brightness ────────────────────────────────────────────────────────────

def score_brightness(img_bgr: np.ndarray, cfg: Config) -> tuple[float, str]:
    """1.0 at the ideal centre of [min, max], falls to 0 at the edges."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())
    mid = (cfg.min_brightness + cfg.max_brightness) / 2
    half = (cfg.max_brightness - cfg.min_brightness) / 2
    score = max(0.0, 1.0 - abs(brightness - mid) / half)
    return score, f"brightness={brightness:.1f}"


# ── Resolution (hard gate, not a weighted component) ─────────────────────

def score_resolution(img_bgr: np.ndarray, cfg: Config) -> tuple[float, str]:
    """Penalise images below the minimum resolution."""
    h, w = img_bgr.shape[:2]
    min_w, min_h = cfg.min_resolution
    score = min(1.0, (w / min_w) * (h / min_h)) ** 0.5
    return score, f"resolution={w}\u00d7{h}"


# ── DeepFace analysis (single call per photo) ────────────────────────────

def analyze_faces(image_path: str) -> dict:
    """Run DeepFace once (thread-safe) and return structured results."""
    with _deepface_lock:
        try:
            results = DeepFace.analyze(
                img_path=image_path,
                actions=["emotion", "age", "gender"],
                enforce_detection=True,
                silent=True,
            )
            if isinstance(results, dict):
                results = [results]
            return {"success": True, "faces": results}
        except Exception as exc:
            return {"success": False, "faces": [], "error": str(exc)}


# ── Face count ────────────────────────────────────────────────────────────

def score_faces(face_data: dict, cfg: Config) -> tuple[float, str]:
    """Score based on face count vs expected range."""
    if not face_data["success"]:
        return 0.0, "No faces detected"
    count = len(face_data["faces"])
    if count < cfg.min_faces:
        return 0.0, f"Too few faces ({count})"
    if count > cfg.max_faces:
        return 0.3, f"Too many faces ({count})"
    return 1.0, f"faces={count}"


# ── Eyes open ─────────────────────────────────────────────────────────────

def score_eyes(face_data: dict) -> tuple[float, str]:
    """
    Estimate whether eyes are open using DeepFace landmarks.
    For full EAR (Eye Aspect Ratio) accuracy consider dlib 68-point or
    MediaPipe FaceMesh — this is a lightweight proxy.
    """
    if not face_data["success"] or not face_data["faces"]:
        return 0.0, "No face for eye check"

    open_scores = []
    for face in face_data["faces"]:
        region = face.get("region", {})
        le = region.get("left_eye")
        re = region.get("right_eye")
        if le and re:
            face_h = region.get("h", 1)
            sep = abs(le[1] - re[1])
            openness = max(0.0, 1.0 - sep / (face_h * 0.3))
            open_scores.append(openness)
        else:
            open_scores.append(0.7)

    score = float(np.mean(open_scores)) if open_scores else 0.5
    return score, f"eyes={score:.2f}"


# ── Emotion ───────────────────────────────────────────────────────────────

def score_emotion(face_data: dict) -> tuple[float, str]:
    """Reward happy / neutral / surprise; penalise angry / sad / fear / disgust."""
    if not face_data["success"] or not face_data["faces"]:
        return 0.5, "emotion: no face"

    POSITIVE = {"happy", "neutral", "surprise"}
    NEGATIVE = {"angry", "sad", "fear", "disgust"}

    scores: list[float] = []
    labels: list[str] = []
    for face in face_data["faces"]:
        labels.append(face.get("dominant_emotion", "neutral"))
        em = face.get("emotion", {})
        pos = sum(em.get(e, 0) for e in POSITIVE)
        neg = sum(em.get(e, 0) for e in NEGATIVE)
        total = pos + neg
        scores.append(pos / total if total > 0 else 0.5)

    s = float(np.mean(scores)) if scores else 0.5
    return s, f"emotion({','.join(labels)})={s:.2f}"


# ── Composition (face centring + head-pose heuristic) ────────────────────

def score_composition(face_data: dict, img_bgr: np.ndarray) -> tuple[float, str]:
    """
    Reward centred, well-sized faces with a soft rule-of-thirds attractor.
    Includes a lightweight head-pose heuristic based on face aspect ratio.
    """
    if not face_data["success"] or not face_data["faces"]:
        return 0.5, "composition: no face"

    img_h, img_w = img_bgr.shape[:2]
    scores: list[float] = []
    for face in face_data["faces"]:
        r = face.get("region", {})
        x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
        if w == 0 or h == 0:
            continue

        # Normalised face centre
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h

        # Distance from ideal position (0.5, 0.4 — slightly above mid)
        dist = ((cx - 0.5) ** 2 + (cy - 0.4) ** 2) ** 0.5
        centre_s = max(0.0, 1.0 - dist * 2)

        # Face area proportion — ideally 5–40 % of frame
        area = (w * h) / (img_w * img_h)
        if 0.05 <= area <= 0.40:
            size_s = 1.0
        elif area < 0.05:
            size_s = area / 0.05
        else:
            size_s = max(0.0, 1.0 - (area - 0.40) / 0.40)

        # Head-pose proxy: a frontal face is roughly square
        aspect = w / max(h, 1)
        pose_s = max(0.0, 1.0 - abs(aspect - 1.0) * 1.5)

        scores.append((centre_s + size_s + pose_s) / 3)

    s = float(np.mean(scores)) if scores else 0.5
    return s, f"composition={s:.2f}"


# ── Noise estimation ─────────────────────────────────────────────────────

def score_noise(gray: np.ndarray) -> tuple[float, str]:
    """
    Median-Laplacian noise estimator (Donoho & Johnstone).
    sigma < 5 → excellent, sigma > 30 → very noisy.
    """
    sigma = float(np.median(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))) / 0.6745
    if sigma < 5:
        s = 1.0
    elif sigma > 30:
        s = 0.0
    else:
        s = 1.0 - (sigma - 5) / 25.0
    return s, f"noise_sigma={sigma:.1f}"


# ── Colour quality (colour-cast + local-contrast entropy) ────────────────

def score_color(img_bgr: np.ndarray) -> tuple[float, str]:
    """Detect colour cast via LAB and score contrast via histogram entropy."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_off = float(lab[:, :, 1].mean()) - 128
    b_off = float(lab[:, :, 2].mean()) - 128
    cast = (a_off ** 2 + b_off ** 2) ** 0.5
    cast_s = max(0.0, 1.0 - cast / 20.0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / max(hist.sum(), 1)
    nz = hist[hist > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    contrast_s = min(1.0, entropy / 7.0)

    s = 0.6 * cast_s + 0.4 * contrast_s
    return s, f"color(cast={cast:.1f}, entropy={entropy:.1f})"


# ── Backlit detection ────────────────────────────────────────────────────

def detect_backlit(img_bgr: np.ndarray, face_data: dict) -> tuple[float, str]:
    """Compare face-region brightness vs overall — penalise silhouettes."""
    faces = face_data.get("faces", [])
    if not faces:
        return 1.0, "backlit: no face"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    overall = float(gray.mean())
    if overall == 0:
        return 1.0, "backlit: dark frame"

    ih, iw = gray.shape
    face_vals: list[float] = []
    for f in faces:
        r = f.get("region", {})
        x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
        if w > 0 and h > 0:
            crop = gray[max(0, y):min(ih, y + h), max(0, x):min(iw, x + w)]
            if crop.size:
                face_vals.append(float(crop.mean()))

    if not face_vals:
        return 1.0, "backlit: no measurable face"

    ratio = float(np.mean(face_vals)) / max(overall, 1)
    if ratio > 0.7:
        s = 1.0
    elif ratio > 0.4:
        s = (ratio - 0.4) / 0.3
    else:
        s = 0.0
    return s, f"backlit(ratio={ratio:.2f})"


# ── Red-eye detection ────────────────────────────────────────────────────

def detect_red_eye(img_bgr: np.ndarray, face_data: dict) -> tuple[float, str]:
    """Scan eye regions for saturated-red blobs."""
    faces = face_data.get("faces", [])
    if not faces:
        return 1.0, "red_eye: no face"

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hits = 0
    for face in faces:
        region = face.get("region", {})
        for key in ("left_eye", "right_eye"):
            eye = region.get(key)
            if eye is None:
                continue
            ex, ey = int(eye[0]), int(eye[1])
            r = max(10, region.get("w", 100) // 10)
            y1 = max(0, ey - r)
            y2 = min(img_bgr.shape[0], ey + r)
            x1 = max(0, ex - r)
            x2 = min(img_bgr.shape[1], ex + r)
            roi = hsv[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            m1 = cv2.inRange(roi, np.array([0, 80, 50]), np.array([10, 255, 255]))
            m2 = cv2.inRange(roi, np.array([160, 80, 50]), np.array([180, 255, 255]))
            red_ratio = float((m1 | m2).sum()) / (255 * max(roi.size // 3, 1))
            if red_ratio > 0.15:
                hits += 1

    if hits:
        return max(0.0, 1.0 - hits * 0.3), f"red_eye: {hits} eye(s)"
    return 1.0, "red_eye: none"


# ── EXIF quality signals ─────────────────────────────────────────────────

def score_exif(exif_data: dict) -> tuple[float, str, list[str]]:
    """Flag high ISO, slow shutter, or flash-fired."""
    warnings: list[str] = []
    penalty = 0.0

    iso = exif_data.get("iso")
    if iso is not None:
        if isinstance(iso, tuple):
            iso = iso[0]
        if isinstance(iso, (int, float)):
            if iso > 3200:
                penalty += 0.3
                warnings.append(f"high_ISO={iso}")
            elif iso > 1600:
                penalty += 0.15
                warnings.append(f"elevated_ISO={iso}")

    shutter = exif_data.get("shutter_speed")
    speed: Optional[float] = None
    if shutter is not None:
        if isinstance(shutter, tuple) and len(shutter) == 2 and shutter[1]:
            speed = shutter[0] / shutter[1]
        elif isinstance(shutter, (int, float)):
            speed = float(shutter)
    if speed is not None and speed > 1 / 30:
        penalty += 0.2
        warnings.append(f"slow_shutter={speed:.3f}s")

    flash = exif_data.get("flash")
    if isinstance(flash, int) and flash & 1:
        penalty += 0.05
        warnings.append("flash_fired")

    score = max(0.0, 1.0 - penalty)
    note = "exif: " + (", ".join(warnings) if warnings else "ok")
    return score, note, warnings


# ===========================================================================
# Duplicate detection — keeps the BEST of each cluster
# ===========================================================================

class DuplicateDetector:
    """Perceptual-hash tracker that selects the highest-scoring photo per
    near-duplicate cluster instead of rejecting on first-seen basis."""

    def __init__(self, threshold: int = 8):
        self.threshold = threshold
        self._entries: list[dict] = []          # {path, hash, score}
        self._lock = threading.Lock()

    def register(self, image_path: str, pil_img: Image.Image, score: int):
        """Register a photo with its hash and score for later clustering."""
        try:
            h = imagehash.phash(pil_img)
        except Exception:
            return
        with self._lock:
            self._entries.append({"path": image_path, "hash": h, "score": score})

    def get_duplicates_to_reject(self) -> set[str]:
        """Return paths that are duplicates of a higher-scoring photo."""
        with self._lock:
            entries = list(self._entries)

        used: set[int] = set()
        reject: set[str] = set()
        for i, a in enumerate(entries):
            if i in used:
                continue
            cluster = [a]
            used.add(i)
            for j, b in enumerate(entries):
                if j in used:
                    continue
                if (a["hash"] - b["hash"]) <= self.threshold:
                    cluster.append(b)
                    used.add(j)
            if len(cluster) > 1:
                best = max(cluster, key=lambda e: e["score"])
                for e in cluster:
                    if e["path"] != best["path"]:
                        reject.add(e["path"])
        return reject


# ===========================================================================
# SQLite result cache
# ===========================================================================

class ResultCache:
    """Cache evaluation results keyed by file-content hash."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(file_hash TEXT PRIMARY KEY, filename TEXT, result TEXT, ts TEXT)"
            )
            self._conn.commit()

    @staticmethod
    def file_hash(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get(self, fhash: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM cache WHERE file_hash=?", (fhash,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, fhash: str, fname: str, result: dict):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                (fhash, fname, json.dumps(result), datetime.now().isoformat()),
            )
            self._conn.commit()

    def close(self):
        self._conn.close()


# ===========================================================================
# Per-photo evaluation
# ===========================================================================

@dataclass
class PhotoResult:
    filename: str
    path: str
    is_good: bool
    total_score: int          # 0–100
    breakdown: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    exif_warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


def evaluate_photo(image_path: str, cfg: Config) -> PhotoResult:
    """Score a single photo across all quality dimensions."""
    filename = Path(image_path).name

    # ── 1. Load & validate (reads image ONCE) ──────────────────────────────
    try:
        pil_img, img_bgr, gray = load_image(image_path)
    except Exception as exc:
        return PhotoResult(filename, image_path, False, 0, error=f"Invalid image: {exc}")

    # ── 2. EXIF metadata ───────────────────────────────────────────────────
    exif_data = read_exif(pil_img)

    # ── 3. Resolution gate (hard filter, not a weighted component) ─────────
    res_s, res_n = score_resolution(img_bgr, cfg)
    if res_s < 0.5:
        return PhotoResult(
            filename, image_path, False, 0,
            notes=[f"Below minimum resolution ({res_n})"],
        )

    # ── 4. DeepFace (one call, serialised across threads) ──────────────────
    face_data = analyze_faces(image_path)

    # ── 5. All scoring components ──────────────────────────────────────────
    sharp_s,  sharp_n  = score_sharpness(gray)
    bright_s, bright_n = score_brightness(img_bgr, cfg)
    noise_s,  noise_n  = score_noise(gray)
    color_s,  color_n  = score_color(img_bgr)
    face_s,   face_n   = score_faces(face_data, cfg)
    eyes_s,   eyes_n   = score_eyes(face_data)
    emo_s,    emo_n    = score_emotion(face_data)
    comp_s,   comp_n   = score_composition(face_data, img_bgr)

    # Penalties (multiplicative — drags total down independent of weights)
    redeye_s,  redeye_n  = detect_red_eye(img_bgr, face_data)
    backlit_s, backlit_n = detect_backlit(img_bgr, face_data)
    exif_s, exif_n, exif_warns = score_exif(exif_data)

    penalty = redeye_s * backlit_s * exif_s

    breakdown = {
        "sharpness":   round(sharp_s  * cfg.weight_sharpness),
        "brightness":  round(bright_s * cfg.weight_brightness),
        "face":        round(face_s   * cfg.weight_face),
        "eyes":        round(eyes_s   * cfg.weight_eyes),
        "composition": round(comp_s   * cfg.weight_composition),
        "emotion":     round(emo_s    * cfg.weight_emotion),
        "noise":       round(noise_s  * cfg.weight_noise),
        "color":       round(color_s  * cfg.weight_color),
    }
    raw_total = sum(breakdown.values())
    total = max(0, min(100, round(raw_total * penalty)))

    notes = [
        sharp_n, bright_n, noise_n, color_n,
        face_n, eyes_n, emo_n, comp_n,
        redeye_n, backlit_n, exif_n,
    ]

    return PhotoResult(
        filename=filename,
        path=image_path,
        is_good=total >= cfg.min_score,
        total_score=total,
        breakdown=breakdown,
        notes=notes,
        exif_warnings=exif_warns,
    )


# ===========================================================================
# Bad-photo sub-folder classification
# ===========================================================================

BAD_SUBFOLDERS = {
    "duplicate":      "duplicate",
    "low_resolution": "low_resolution",
    "blurry":         "blurry",
    "poor_lighting":  "poor_lighting",
    "no_face":        "no_face",
    "eyes_closed":    "eyes_closed",
    "noisy":          "noisy",
    "red_eye":        "red_eye",
    "color_cast":     "color_cast",
    "corrupt":        "corrupt",
    "low_score":      "low_score",
}


def classify_rejection_reason(result: PhotoResult, cfg: Config) -> str:
    """Determine the primary rejection reason → sub-folder name."""
    if result.error:
        return "corrupt"

    notes_joined = " | ".join(result.notes or [])

    # Explicit hard-filter reasons
    if "Duplicate" in notes_joined:
        return "duplicate"
    if "Below minimum resolution" in notes_joined:
        return "low_resolution"
    if "red_eye:" in notes_joined and "none" not in notes_joined:
        return "red_eye"

    if not result.breakdown:
        return "low_score"

    # Find the weakest component (lowest achieved / max-weight ratio)
    weight_map = {
        "sharpness":   cfg.weight_sharpness,
        "brightness":  cfg.weight_brightness,
        "face":        cfg.weight_face,
        "eyes":        cfg.weight_eyes,
        "composition": cfg.weight_composition,
        "emotion":     cfg.weight_emotion,
        "noise":       cfg.weight_noise,
        "color":       cfg.weight_color,
    }
    folder_map = {
        "sharpness":   "blurry",
        "brightness":  "poor_lighting",
        "face":        "no_face",
        "eyes":        "eyes_closed",
        "composition": "low_score",
        "emotion":     "low_score",
        "noise":       "noisy",
        "color":       "color_cast",
    }

    worst, worst_ratio = None, 1.0
    for comp, achieved in result.breakdown.items():
        mw = weight_map.get(comp, 0)
        if mw == 0:
            continue
        ratio = achieved / mw
        if ratio < worst_ratio:
            worst_ratio = ratio
            worst = comp

    return folder_map.get(worst, "low_score") if worst else "low_score"


# ===========================================================================
# HTML gallery report
# ===========================================================================

def generate_html_report(
    results: list[PhotoResult], cfg: Config, good: int, bad: int,
):
    """Write a self-contained HTML gallery with scores and filter buttons."""
    sorted_results = sorted(results, key=lambda r: r.total_score, reverse=True)

    cards_html = ""
    for r in sorted_results:
        css_class = "good" if r.is_good else "bad"
        tag = "GOOD" if r.is_good else "BAD"
        tag_css = "tag-good" if r.is_good else "tag-bad"

        # Relative path to image in the output folder
        if r.is_good:
            img_rel = f"{cfg.good_folder}/{r.filename}"
        else:
            reason = classify_rejection_reason(r, cfg)
            img_rel = f"{cfg.bad_folder}/{reason}/{r.filename}"

        breakdown_lines = "".join(
            f'<div class="bar-row">'
            f'  <span class="bar-label">{k}</span>'
            f'  <div class="bar"><div class="bar-fill {css_class}" '
            f'       style="width:{v}%"></div></div>'
            f'  <span class="bar-val">{v}</span>'
            f"</div>"
            for k, v in r.breakdown.items()
        )

        notes_str = "<br>".join(r.notes) if r.notes else ""
        error_str = f'<div class="error">{r.error}</div>' if r.error else ""

        cards_html += f"""
        <div class="card" data-good="{1 if r.is_good else 0}">
          <img src="{img_rel}" alt="{r.filename}" loading="lazy"
               onerror="this.style.display='none'">
          <div class="card-body">
            <div class="card-title">{r.filename}</div>
            <span class="tag {tag_css}">{tag}</span>
            <div class="score {css_class}">{r.total_score}</div>
            {breakdown_lines}
            {error_str}
            <details><summary>Notes</summary>
              <p class="notes">{notes_str}</p>
            </details>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Photo Classification Report</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#0f0f1a;color:#ddd}}
h1{{text-align:center;margin-bottom:4px}}
.subtitle{{text-align:center;color:#888;margin-bottom:24px}}
.summary{{display:flex;gap:16px;justify-content:center;margin-bottom:28px;flex-wrap:wrap}}
.summary-card{{background:#1a1a2e;padding:18px 32px;border-radius:12px;text-align:center}}
.summary-card h2{{margin:0;font-size:2.2em}}
.good{{color:#4ecca3}}.bad{{color:#e23e57}}
.filter-bar{{text-align:center;margin-bottom:20px}}
.filter-bar button{{padding:8px 18px;margin:4px;border:none;border-radius:6px;
  cursor:pointer;background:#1a1a2e;color:#ccc;font-size:.95em}}
.filter-bar button.active{{background:#4ecca3;color:#0f0f1a;font-weight:600}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}}
.card{{background:#1a1a2e;border-radius:10px;overflow:hidden;transition:transform .15s}}
.card:hover{{transform:scale(1.02)}}
.card img{{width:100%;height:200px;object-fit:cover}}
.card-body{{padding:12px}}
.card-title{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.score{{font-size:1.6em;font-weight:700;margin:4px 0}}
.tag{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.78em;margin:4px 4px 4px 0}}
.tag-good{{background:#14332a;color:#4ecca3}}.tag-bad{{background:#3d0f0f;color:#e23e57}}
.bar-row{{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:.8em}}
.bar-label{{width:80px;text-align:right;color:#999}}
.bar{{flex:1;height:6px;background:#333;border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px}}
.bar-fill.good{{background:#4ecca3}}.bar-fill.bad{{background:#e23e57}}
.bar-val{{width:24px;color:#aaa}}
.notes{{font-size:.8em;color:#888;word-break:break-all}}
.error{{color:#e23e57;font-size:.85em;margin-top:6px}}
details summary{{cursor:pointer;color:#666;font-size:.85em}}
</style></head><body>
<h1>&#128248; Photo Classification Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="summary">
  <div class="summary-card"><h2 class="good">{good}</h2>Good</div>
  <div class="summary-card"><h2 class="bad">{bad}</h2>Bad</div>
  <div class="summary-card"><h2>{len(results)}</h2>Total</div>
</div>
<div class="filter-bar">
  <button class="active" onclick="filterCards('all',this)">All</button>
  <button onclick="filterCards('good',this)">Good</button>
  <button onclick="filterCards('bad',this)">Bad</button>
</div>
<div class="gallery" id="gallery">{cards_html}
</div>
<script>
function filterCards(f, btn) {{
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    const isGood = c.dataset.good === '1';
    c.style.display = (f === 'all' || (f === 'good' && isGood) || (f === 'bad' && !isGood)) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

    Path(cfg.html_report_file).write_text(html, encoding="utf-8")
    log.info("HTML report -> '%s'", cfg.html_report_file)


# ===========================================================================
# Main pipeline
# ===========================================================================

def process_photos(
    cfg: Config,
    dry_run: bool = False,
    save_report: bool = False,
    html_report: bool = False,
):
    cfg.apply_profile()

    input_path = Path(cfg.input_folder)
    if not input_path.exists():
        log.error("Input folder '%s' not found.", cfg.input_folder)
        return

    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in cfg.extensions
    )
    if not image_files:
        log.warning("No images found in '%s'.", cfg.input_folder)
        return

    log.info(
        "Found %d image(s) | profile=%s | workers=%d | device=%s",
        len(image_files), cfg.profile, cfg.workers, cfg.device,
    )

    # ── Optional result cache ──────────────────────────────────────────────
    cache: Optional[ResultCache] = (
        ResultCache(cfg.cache_db) if cfg.enable_cache else None
    )

    # ── Phase 1: evaluate every photo ──────────────────────────────────────
    path_to_result: dict[str, PhotoResult] = {}

    def _eval_or_cache(fpath: str) -> PhotoResult:
        fhash = None
        if cache:
            fhash = ResultCache.file_hash(fpath)
            cached = cache.get(fhash)
            if cached:
                return PhotoResult(**cached)
        result = evaluate_photo(fpath, cfg)
        if cache and fhash:
            cache.put(fhash, result.filename, {
                "filename": result.filename,
                "path": result.path,
                "is_good": result.is_good,
                "total_score": result.total_score,
                "breakdown": result.breakdown,
                "notes": result.notes,
                "exif_warnings": result.exif_warnings,
                "error": result.error,
            })
        return result

    progress = (
        tqdm(total=len(image_files), desc="Scoring", unit="img")
        if HAS_TQDM else None
    )

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_eval_or_cache, str(f)): f for f in image_files
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = PhotoResult(src.name, str(src), False, 0, error=str(exc))

            path_to_result[str(src)] = result

            status = "GOOD" if result.is_good else " BAD"
            log.info(
                "%s  score=%d  %s", status, result.total_score, result.filename,
            )
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    # ── Phase 2: best-of-duplicate-cluster ─────────────────────────────────
    log.info("Resolving duplicate clusters ...")
    dedup = DuplicateDetector(threshold=cfg.duplicate_threshold)
    for path, result in path_to_result.items():
        if result.error is None:
            try:
                pil_img = Image.open(path)
                pil_img = ImageOps.exif_transpose(pil_img) or pil_img
                dedup.register(path, pil_img, result.total_score)
            except Exception:
                pass

    dup_rejects = dedup.get_duplicates_to_reject()
    for path in dup_rejects:
        r = path_to_result[path]
        r.is_good = False
        r.notes.insert(0, "Duplicate (better version kept)")
        r.total_score = 0

    # ── Phase 3: copy files to folders (bad → sub-folders by reason) ───────
    results = list(path_to_result.values())
    good, bad = 0, 0

    if not dry_run:
        Path(cfg.good_folder).mkdir(exist_ok=True)
        Path(cfg.bad_folder).mkdir(exist_ok=True)
        for name in BAD_SUBFOLDERS.values():
            Path(cfg.bad_folder, name).mkdir(exist_ok=True)

    for r in results:
        src = Path(r.path)
        if r.is_good:
            good += 1
            if not dry_run:
                shutil.copy2(src, os.path.join(cfg.good_folder, r.filename))
        else:
            bad += 1
            reason = classify_rejection_reason(r, cfg)
            if not dry_run:
                dest = os.path.join(cfg.bad_folder, reason, r.filename)
                shutil.copy2(src, dest)

    # ── Summary ────────────────────────────────────────────────────────────
    dup_count = len(dup_rejects)
    print(f"\n{'=' * 60}")
    print(f"  Profile         : {cfg.profile}")
    print(f"  Total processed : {len(results)}")
    print(f"  Good photos     : {good}")
    print(f"  Bad  photos     : {bad}  (duplicates removed: {dup_count})")
    if not dry_run:
        print(f"  Good  ->  '{cfg.good_folder}/'")
        print(f"  Bad   ->  '{cfg.bad_folder}/<reason>/'")
        print(f"  Bad sub-folders : {', '.join(sorted(BAD_SUBFOLDERS.values()))}")
    else:
        print("  Dry-run mode -- no files were copied.")
    print(f"{'=' * 60}\n")

    # ── Reports ────────────────────────────────────────────────────────────
    if save_report:
        report = {
            "config": asdict(cfg),
            "summary": {
                "total": len(results), "good": good, "bad": bad,
                "duplicates_removed": dup_count,
            },
            "photos": [
                {
                    "filename": r.filename,
                    "is_good": r.is_good,
                    "score": r.total_score,
                    "breakdown": r.breakdown,
                    "notes": r.notes,
                    "exif_warnings": r.exif_warnings,
                    "error": r.error,
                    "rejection_reason": (
                        classify_rejection_reason(r, cfg) if not r.is_good else None
                    ),
                }
                for r in sorted(results, key=lambda r: r.total_score, reverse=True)
            ],
        }
        with open(cfg.report_file, "w") as fh:
            json.dump(report, fh, indent=2)
        log.info("JSON report -> '%s'", cfg.report_file)

    if html_report:
        generate_html_report(results, cfg, good, bad)

    if cache:
        cache.close()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart photo quality classifier (v2)",
    )
    p.add_argument("--input",       default="input_photos",
                   help="Input folder")
    p.add_argument("--good",        default="good_photos",
                   help="Output folder for good photos")
    p.add_argument("--bad",         default="bad_photos",
                   help="Output: bad photos (sub-folders auto-created)")
    p.add_argument("--min-score",   type=int, default=60,
                   help="Minimum score to be 'good' (0-100)")
    p.add_argument("--workers",     type=int, default=4,
                   help="Parallel workers")
    p.add_argument("--profile",     choices=list(PROFILES.keys()),
                   default="portrait",
                   help="Scoring profile (portrait|group|landscape)")
    p.add_argument("--device",      choices=["cpu", "cuda"], default="cpu",
                   help="Compute device for DeepFace")
    p.add_argument("--dry-run",     action="store_true",
                   help="Analyse only, don't copy files")
    p.add_argument("--report",      action="store_true",
                   help="Write report.json")
    p.add_argument("--html-report", action="store_true",
                   help="Write report.html gallery")
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable SQLite result cache")
    args = p.parse_args()

    cfg = Config(
        input_folder=args.input,
        good_folder=args.good,
        bad_folder=args.bad,
        min_score=args.min_score,
        workers=args.workers,
        profile=args.profile,
        device=args.device,
        enable_cache=not args.no_cache,
    )
    return cfg, args.dry_run, args.report, args.html_report


if __name__ == "__main__":
    cfg, dry_run, save_report, html_report = parse_args()
    process_photos(cfg, dry_run=dry_run, save_report=save_report, html_report=html_report)
