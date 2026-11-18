"""
SmartElec - Voter Registration Script
Captures voter facial biometrics via webcam or mock generator, extracts 50x50 normalized
feature vectors, and persists records to faces_data.pkl, names.pkl, and voters.csv.
"""

import sys
import time
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from smartelec.config import (
    NUM_SAMPLES_PER_VOTER,
    FACE_IMG_SIZE,
    PROJECT_ROOT
)
from smartelec.face_detector import FaceDetector
from smartelec.voter_registry import VoterRegistry
from smartelec.audio_feedback import AudioFeedback


def parse_arguments():
    parser = argparse.ArgumentParser(description="SmartElec Voter Biometric Registration")
    parser.add_argument("--id", type=str, help="Unique Voter ID (e.g. VOTER-01)")
    parser.add_argument("--name", type=str, help="Full Name of the Voter")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES_PER_VOTER, help="Number of face snapshots to capture")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic biometric data for testing without camera")
    return parser.parse_args()


def generate_mock_face_samples(num_samples: int, seed_id: str) -> np.ndarray:
    """Generates synthetic distinctive face feature vectors for headless testing."""
    # Deterministic pseudo-random seed based on voter ID
    seed = sum(ord(c) for c in seed_id) % 10000
    rng = np.random.default_rng(seed)
    base_face = rng.normal(loc=128.0, scale=35.0, size=(FACE_IMG_SIZE[0] * FACE_IMG_SIZE[1],)).astype(np.float32)

    samples = []
    for _ in range(num_samples):
        # Add slight natural variation (noise simulating lighting / micro-expressions)
        noise = rng.normal(loc=0.0, scale=3.0, size=base_face.shape).astype(np.float32)
        sample = np.clip(base_face + noise, 0, 255)
        samples.append(sample)

    return np.array(samples, dtype=np.float32)


def main():
    args = parse_arguments()
    registry = VoterRegistry()
    audio = AudioFeedback()

    print("=" * 65)
    print("      SMARTELEC - BIOMETRIC VOTER REGISTRATION PORTAL")
    print("=" * 65)

    voter_id = args.id
    if not voter_id:
        voter_id = input("Enter Voter ID (e.g. VOTER-101): ").strip()

    voter_name = args.name
    if not voter_name:
        voter_name = input("Enter Voter Full Name: ").strip()

    if not voter_id or not voter_name:
        print("[ERROR] Voter ID and Full Name are mandatory.")
        sys.exit(1)

    voter_id = voter_id.upper()
    if registry.is_voter_id_registered(voter_id):
        print(f"[ERROR] Voter ID '{voter_id}' is already registered in the system.")
        audio.play_warning_beep()
        sys.exit(1)

    total_samples = args.samples
    print(f"\n[INFO] Initializing capture for {voter_name} ({voter_id}). Target: {total_samples} samples.")

    # Mock mode branch (works without camera)
    if args.mock or cv2 is None:
        if args.mock:
            print("[INFO] Running in mock/simulation mode (generating synthetic biometrics)...")
        else:
            print("[WARN] OpenCV GUI not available; falling back to simulated biometric capture.")

        samples = generate_mock_face_samples(total_samples, voter_id)
        success, msg = registry.save_voter_biometrics(voter_id, voter_name, samples)
        if success:
            print(f"[SUCCESS] {msg}")
            audio.play_success_beep()
            audio.speak(f"Registration successful for {voter_name}.")
        else:
            print(f"[ERROR] {msg}")
        return

    # Live Camera Capture
    detector = FaceDetector()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[WARN] Unable to open camera index {args.camera}. Falling back to simulation.")
        samples = generate_mock_face_samples(total_samples, voter_id)
        success, msg = registry.save_voter_biometrics(voter_id, voter_name, samples)
        if success:
            print(f"[SUCCESS] {msg}")
            audio.play_success_beep()
            audio.speak(f"Registration complete for {voter_name}.")
        return

    audio.speak("Please look at the camera for biometric facial scanning.")
    captured_vectors = []
    frame_count = 0

    print("[INFO] Camera opened. Press 'q' at any time to cancel.")

    while len(captured_vectors) < total_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame_count += 1
        # Flip frame horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)
        faces = detector.detect_faces(frame)

        status_text = f"Capturing: {len(captured_vectors)}/{total_samples}"

        if len(faces) == 1:
            bbox = faces[0]
            detector.draw_face_hud(frame, bbox, f"{voter_name} ({len(captured_vectors)}/{total_samples})", status="SCANNING")

            # Capture face feature every 2 frames for variety
            if frame_count % 2 == 0:
                features = detector.extract_face_features(frame, bbox)
                captured_vectors.append(features)
        elif len(faces) > 1:
            # Multiple faces detected
            status_text = "MULTIPLE FACES DETECTED - Ensure only 1 person is in frame"
            for bbox in faces:
                detector.draw_face_hud(frame, bbox, "EXTRA FACE", status="UNREGISTERED")
        else:
            status_text = "NO FACE DETECTED - Center your face in the camera"

        # Draw Progress Bar HUD
        progress = len(captured_vectors) / total_samples
        bar_w = 400
        bar_h = 16
        bar_x = int((frame.shape[1] - bar_w) / 2)
        bar_y = frame.shape[0] - 40

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        cv2.putText(frame, status_text, (bar_x, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("SmartElec - Biometric Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Registration canceled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_vectors) >= total_samples:
        face_array = np.array(captured_vectors, dtype=np.float32)
        success, msg = registry.save_voter_biometrics(voter_id, voter_name, face_array)
        if success:
            print(f"\n[SUCCESS] {msg}")
            audio.play_success_beep()
            audio.speak(f"Registration successful for {voter_name}.")
        else:
            print(f"\n[ERROR] {msg}")
    else:
        print("[WARN] Incomplete face samples captured. Aborting.")


if __name__ == "__main__":
    main()
