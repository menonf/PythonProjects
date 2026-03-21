import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import imagehash

# Configuration
INPUT_FOLDER = "input_photos"
GOOD_PHOTOS_FOLDER = "good_photos"
BAD_PHOTOS_FOLDER = "bad_photos"

# Thresholds (adjust as needed)
MIN_FACE_CONFIDENCE = 0.9
MIN_IMAGE_SHARPNESS = 100.0
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 220
MIN_FACES = 1  # Minimum number of faces required


def setup_folders():
    """Create output folders if they don't exist."""
    Path(GOOD_PHOTOS_FOLDER).mkdir(exist_ok=True)
    Path(BAD_PHOTOS_FOLDER).mkdir(exist_ok=True)


def check_sharpness(image_path: str) -> float:
    """Check image sharpness using Laplacian variance."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_brightness(image_path: str) -> float:
    """Check average image brightness."""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].mean()


def check_faces(image_path: str) -> dict:
    """
    Detect faces and analyze emotions/attributes using DeepFace.
    Returns face analysis results.
    """
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion", "age"],
            enforce_detection=True,
            silent=True,
        )
        return {"success": True, "faces": results, "face_count": len(results)}
    except Exception:
        return {"success": False, "faces": [], "face_count": 0}


def is_duplicate(image_path: str, seen_hashes: set) -> bool:
    """Check if image is a duplicate using perceptual hashing."""
    try:
        img = Image.open(image_path)
        img_hash = imagehash.phash(img)
        if img_hash in seen_hashes:
            return True
        seen_hashes.add(img_hash)
        return False
    except Exception:
        return False


def evaluate_photo(image_path: str, seen_hashes: set) -> tuple[bool, list[str]]:
    """
    Evaluate a photo and return (is_good, reasons_if_bad).
    """
    reasons = []

    # Check if file is a valid image
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception:
        reasons.append("Invalid or corrupted image")
        return False, reasons

    # Check for duplicates
    if is_duplicate(image_path, seen_hashes):
        reasons.append("Duplicate image")
        return False, reasons

    # Check sharpness (blurry photos)
    sharpness = check_sharpness(image_path)
    if sharpness < MIN_IMAGE_SHARPNESS:
        reasons.append(f"Image too blurry (sharpness: {sharpness:.1f})")

    # Check brightness (too dark or too bright)
    brightness = check_brightness(image_path)
    if brightness < MIN_BRIGHTNESS:
        reasons.append(f"Image too dark (brightness: {brightness:.1f})")
    elif brightness > MAX_BRIGHTNESS:
        reasons.append(f"Image too bright/overexposed (brightness: {brightness:.1f})")

    # Check for faces
    face_results = check_faces(image_path)
    if not face_results["success"] or face_results["face_count"] < MIN_FACES:
        reasons.append("No clear faces detected")
    else:
        print(f"  Found {face_results['face_count']} face(s)")

    is_good = len(reasons) == 0
    return is_good, reasons


def process_photos():
    """Main function to process all photos in the input folder."""
    setup_folders()

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Get all image files
    image_files = [
        f
        for f in os.listdir(INPUT_FOLDER)
        if Path(f).suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}' folder.")
        return

    print(f"Found {len(image_files)} images to process...\n")

    seen_hashes = set()
    good_count = 0
    bad_count = 0

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(INPUT_FOLDER, filename)
        print(f"[{idx}/{len(image_files)}] Processing: {filename}")

        is_good, reasons = evaluate_photo(image_path, seen_hashes)

        if is_good:
            shutil.copy2(image_path, os.path.join(GOOD_PHOTOS_FOLDER, filename))
            print(f"  ✅ GOOD photo")
            good_count += 1
        else:
            shutil.copy2(image_path, os.path.join(BAD_PHOTOS_FOLDER, filename))
            print(f"  ❌ BAD photo - Reasons: {', '.join(reasons)}")
            bad_count += 1

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"✅ Good photos: {good_count}")
    print(f"❌ Bad photos:  {bad_count}")
    print(f"Good photos saved to: '{GOOD_PHOTOS_FOLDER}'")
    print(f"Bad photos saved to:  '{BAD_PHOTOS_FOLDER}'")


if __name__ == "__main__":
    process_photos()
