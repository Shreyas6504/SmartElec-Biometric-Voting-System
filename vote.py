"""
SmartElec - Real-Time Biometric Voting Terminal
Integrates OpenCV Face Detection, K-Nearest Neighbors Classifier (NumPy or Scikit-Learn),
eligibility checking, ballot casting, and duplicate-vote prevention with CSV logging.
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
    CANDIDATES,
    KNN_K_NEIGHBORS,
    UNKNOWN_DISTANCE_THRESHOLD
)
from smartelec.face_detector import FaceDetector
from smartelec.voter_registry import VoterRegistry
from smartelec.voting_manager import VotingManager
from smartelec.knn_classifier import FaceKNNClassifier
from smartelec.audio_feedback import AudioFeedback


def parse_arguments():
    parser = argparse.ArgumentParser(description="SmartElec Real-Time Voting Terminal")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--engine", type=str, choices=["numpy", "sklearn"], default="numpy",
                        help="KNN calculation engine: 'numpy' (custom implementation) or 'sklearn' (default: numpy)")
    parser.add_argument("--k", type=int, default=KNN_K_NEIGHBORS, help="Number of nearest neighbors (k)")
    parser.add_argument("--mock", action="store_true", help="Run interactive terminal simulation without physical camera")
    return parser.parse_args()


def draw_terminal_hud(frame, detector, faces, current_status, current_name, candidate_keys, message_banner=None):
    """
    Renders high-visibility election UI overlays directly on the OpenCV video feed.
    """
    h, w = frame.shape[:2]

    # 1. Top Header Banner
    header_h = 50
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (20, 24, 30), -1)
    cv2.addWeighted(header_overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "SMARTELEC: SECURE BIOMETRIC VOTING TERMINAL", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    # 2. Status Bar Below Header
    status_colors = {
        "ELIGIBLE": (0, 200, 0),       # Green
        "ALREADY_VOTED": (0, 140, 255), # Amber / Orange
        "UNREGISTERED": (0, 0, 230),    # Red
        "SCANNING": (200, 200, 0)       # Cyan
    }
    bar_color = status_colors.get(current_status, (150, 150, 150))
    cv2.rectangle(frame, (0, header_h), (w, header_h + 35), bar_color, -1)

    if current_status == "ELIGIBLE":
        status_msg = f"VOTER IDENTIFIED: {current_name.upper()} | STATUS: ELIGIBLE | CAST VOTE [1-4]"
        text_color = (0, 0, 0)
    elif current_status == "ALREADY_VOTED":
        status_msg = f"SECURITY ALERT: {current_name.upper()} HAS ALREADY VOTED | FRAUD PREVENTED"
        text_color = (0, 0, 0)
    elif current_status == "UNREGISTERED":
        status_msg = "ACCESS DENIED: UNREGISTERED VOTER FACE | CONTACT ELECTION OFFICIAL"
        text_color = (255, 255, 255)
    else:
        status_msg = "SCANNING FOR VOTER FACE... PLEASE STAND DIRECTLY BEFORE THE CAMERA"
        text_color = (0, 0, 0)

    cv2.putText(frame, status_msg, (20, header_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA)

    # 3. Bottom Candidate Ballot Panel
    panel_h = 105
    panel_y = h - panel_h
    panel_overlay = frame.copy()
    cv2.rectangle(panel_overlay, (0, panel_y), (w, h), (15, 18, 22), -1)
    cv2.addWeighted(panel_overlay, 0.90, frame, 0.10, 0, frame)

    cv2.line(frame, (0, panel_y), (w, panel_y), (0, 220, 255), 2)
    cv2.putText(frame, "OFFICIAL BALLOT CHOICES (Press key on keyboard to vote):", (20, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    col1_x = 25
    col2_x = int(w / 2) + 15
    y_r1 = panel_y + 52
    y_r2 = panel_y + 85

    cand = CANDIDATES
    c1 = f"[1] {cand['1']['name']} ({cand['1']['party']})"
    c2 = f"[2] {cand['2']['name']} ({cand['2']['party']})"
    c3 = f"[3] {cand['3']['name']} ({cand['3']['party']})"
    c4 = f"[4] {cand['4']['name']} ({cand['4']['party']})"

    cv2.putText(frame, c1, (col1_x, y_r1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, c2, (col2_x, y_r1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, c3, (col1_x, y_r2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, c4, (col2_x, y_r2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Success / Pop-up Splash Banner if active
    if message_banner:
        banner_w = int(w * 0.8)
        banner_h = 70
        bx = int((w - banner_w) / 2)
        by = int((h - banner_h) / 2)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 180, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (255, 255, 255), 2)
        cv2.putText(frame, message_banner, (bx + 20, by + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def run_interactive_simulation(registry: VoterRegistry, voting_mgr: VotingManager, knn: FaceKNNClassifier, audio: AudioFeedback):
    """
    Console simulation mode when camera hardware is absent or testing in headless environments.
    """
    print("\n" + "=" * 65)
    print("      SMARTELEC - INTERACTIVE VOTING CONSOLE SIMULATION")
    print("=" * 65)

    voters = registry.get_all_voters()
    if not voters:
        print("[ERROR] No voters registered. Please run `python register_voter.py` first.")
        return

    print("\nRegistered Voters in System:")
    for idx, v in enumerate(voters, start=1):
        has_voted = voting_mgr.has_voter_already_voted(v["VoterID"])
        voted_badge = "[ALREADY VOTED]" if has_voted else "[ELIGIBLE]"
        print(f"  {idx}. {v['FullName']} ({v['VoterID']}) - {voted_badge}")

    print("\nSelect a voter to simulate standing in front of the camera:")
    choice = input("Enter Voter Number (or 'q' to quit): ").strip()
    if choice.lower() == 'q' or not choice.isdigit():
        return

    selected_idx = int(choice) - 1
    if selected_idx < 0 or selected_idx >= len(voters):
        print("[ERROR] Invalid choice.")
        return

    target_voter = voters[selected_idx]
    voter_id = target_voter["VoterID"]
    voter_name = target_voter["FullName"]

    print(f"\n[BIOMETRIC SCAN]: Face matching against trained KNN model...")
    # Simulate recognition
    tag = f"{voter_id}:{voter_name}"
    status, vid, vname = voting_mgr.check_voter_eligibility(tag)

    print(f"[RECOGNIZED]: {voter_name} ({voter_id})")
    print(f"[STATUS]: {status}")

    if status == "ALREADY_VOTED":
        print(f"[ALERT] Voter '{voter_name}' has already cast their ballot. Fraud prevented!")
        audio.play_warning_beep()
        audio.speak(f"Alert. {voter_name} has already voted.")
        return
    elif status == "UNREGISTERED":
        print("[ALERT] Unregistered face. Access denied.")
        audio.play_warning_beep()
        return

    audio.speak(f"Welcome {voter_name}. Please cast your vote.")
    print("\nCandidates on Ballot:")
    for k, cand in CANDIDATES.items():
        print(f"  [{k}] {cand['name']} - {cand['party']} {cand.get('symbol', '')}")

    vote_key = input("\nEnter Candidate Choice (1-4): ").strip()
    success, msg, record = voting_mgr.cast_vote(voter_id, vote_key)

    if success:
        print(f"\n[SUCCESS]: {msg}")
        print(f"[AUDIT HASH]: {record['AuditHash'][:32]}...")
        audio.play_success_beep()
        audio.speak(f"Vote successfully recorded. Thank you {voter_name}!")
    else:
        print(f"\n[ERROR]: {msg}")
        audio.play_warning_beep()


def main():
    args = parse_arguments()
    registry = VoterRegistry()
    voting_mgr = VotingManager(registry=registry)
    audio = AudioFeedback()

    print("=" * 65)
    print("      SMARTELEC - REAL-TIME BIOMETRIC VOTING TERMINAL")
    print("=" * 65)
    print(f"[CONFIG] KNN Engine: {args.engine.upper()} | Neighbors (k): {args.k}")

    # 1. Load Training Dataset
    faces, names = registry.load_dataset()
    if faces is None or names is None or len(faces) == 0:
        print("\n[ERROR] No voter biometric dataset found.")
        print("Please register at least one voter first using:")
        print("  python register_voter.py --id VOTER-01 --name \"John Doe\"")
        sys.exit(1)

    print(f"[INFO] Loaded biometric dataset: {len(faces)} face samples across {len(set(names))} profiles.")

    # 2. Train KNN Model
    print(f"[INFO] Fitting KNN classifier using {args.engine} engine...")
    knn = FaceKNNClassifier(n_neighbors=args.k, engine=args.engine, distance_threshold=UNKNOWN_DISTANCE_THRESHOLD)
    knn.fit(faces, names)
    print("[SUCCESS] Classifier trained and ready for real-time inference.")

    # Check mock / headless mode
    if args.mock or cv2 is None:
        run_interactive_simulation(registry, voting_mgr, knn, audio)
        return

    # 3. Initialize Camera
    detector = FaceDetector()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[WARN] Camera {args.camera} not accessible. Launching interactive simulation mode.")
        run_interactive_simulation(registry, voting_mgr, knn, audio)
        return

    print("\n[READY] Voting Terminal Active.")
    print("  - Position face towards camera.")
    print("  - When status is ELIGIBLE, press [1], [2], [3], or [4] to cast vote.")
    print("  - Press 'q' to terminate voting terminal.\n")

    audio.speak("SmartElec voting terminal is now active. Please face the camera.")

    active_voter_id = ""
    active_voter_name = ""
    current_status = "SCANNING"
    splash_banner = None
    splash_timer = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera stream interrupted.")
            break

        frame = cv2.flip(frame, 1)
        faces_detected = detector.detect_faces(frame)

        if len(faces_detected) == 1:
            bbox = faces_detected[0]
            face_feature = detector.extract_face_features(frame, bbox)

            # Predict identity with KNN
            predicted_label, distance, is_confident = knn.predict(face_feature)

            if is_confident and predicted_label not in ["Unknown", "Unregistered"]:
                status_code, vid, vname = voting_mgr.check_voter_eligibility(predicted_label)
                active_voter_id = vid
                active_voter_name = vname
                current_status = status_code
                hud_label = f"{vname} ({distance:.0f})"
            else:
                active_voter_id = ""
                active_voter_name = "Unregistered"
                current_status = "UNREGISTERED"
                hud_label = f"Unregistered ({distance:.0f})"

            detector.draw_face_hud(frame, bbox, hud_label, status=current_status)

        elif len(faces_detected) > 1:
            current_status = "UNREGISTERED"
            active_voter_id = ""
            active_voter_name = "Multiple Faces"
            for bbox in faces_detected:
                detector.draw_face_hud(frame, bbox, "FLAGGED", status="UNREGISTERED")
        else:
            current_status = "SCANNING"
            active_voter_id = ""
            active_voter_name = ""

        # Handle Splash Message Timer
        if splash_banner and time.time() > splash_timer:
            splash_banner = None

        # Draw GUI Terminal Overlays
        frame = draw_terminal_hud(
            frame, detector, faces_detected, current_status, active_voter_name,
            CANDIDATES, message_banner=splash_banner
        )

        cv2.imshow("SmartElec - Biometric Election Terminal", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n[INFO] Closing voting terminal...")
            break

        # Check Voting Keypress (1, 2, 3, 4)
        char_key = chr(key) if 0 <= key <= 255 else ""
        if char_key in CANDIDATES:
            if current_status == "ELIGIBLE" and active_voter_id:
                success, msg, record = voting_mgr.cast_vote(active_voter_id, char_key)
                if success:
                    cand_name = record["CandidateName"]
                    print(f"\n[VOTE CAST] {active_voter_name} voted for {cand_name} (Hash: {record['AuditHash'][:16]}...)")
                    audio.play_success_beep()
                    audio.speak(f"Vote recorded for {cand_name}. Thank you {active_voter_name}.")
                    splash_banner = f"VOTE CONFIRMED: {cand_name} | THANK YOU!"
                    splash_timer = time.time() + 2.5
                    current_status = "ALREADY_VOTED"
                else:
                    print(f"\n[ERROR] {msg}")
                    audio.play_warning_beep()
            elif current_status == "ALREADY_VOTED":
                print(f"\n[ALERT] Voter '{active_voter_name}' has already cast a ballot!")
                audio.play_warning_beep()
                audio.speak("Alert. You have already cast your vote.")
                splash_banner = "ALERT: YOU HAVE ALREADY VOTED!"
                splash_timer = time.time() + 2.0
            elif current_status == "UNREGISTERED":
                print("\n[DENIED] Unregistered voter attempt.")
                audio.play_warning_beep()
                splash_banner = "ACCESS DENIED: UNREGISTERED VOTER"
                splash_timer = time.time() + 2.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
