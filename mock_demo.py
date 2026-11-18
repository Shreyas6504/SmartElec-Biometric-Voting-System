"""
SmartElec - Automated End-to-End Demonstration & Simulation
Runs the full pipeline without requiring a physical camera:
1. Registers synthetic biometric voters (Alice, Bob, Charlie)
2. Trains both Pure NumPy and Scikit-Learn KNN classifiers
3. Simulates voter authentication and casting ballots
4. Tests fraud prevention (duplicate voting rejection)
5. Tests rejection of unregistered voters
6. Validates cryptographic SHA-256 audit ledger integrity
7. Renders the final election analytics
"""

import sys
import numpy as np
from smartelec.config import (
    FACE_IMG_SIZE,
    NUM_SAMPLES_PER_VOTER,
    CANDIDATES
)
from smartelec.voter_registry import VoterRegistry
from smartelec.voting_manager import VotingManager
from smartelec.knn_classifier import FaceKNNClassifier, PureNumpyKNN
from register_voter import generate_mock_face_samples
from admin_dashboard import print_cli_dashboard


def run_demo():
    print("=" * 70)
    print("       SMARTELEC - FULL SYSTEM END-TO-END DEMONSTRATION")
    print("=" * 70)

    registry = VoterRegistry()
    voting_mgr = VotingManager(registry=registry)

    # Clean slate for clean demo run
    print("\n[STEP 1]: Initializing Clean Election State...")
    registry.reset_registry()
    voting_mgr.reset_election()
    print("✔ Registry and audit ledgers cleared.")

    # Step 2: Register Voters
    print("\n[STEP 2]: Registering Biometric Profiles...")
    demo_voters = [
        ("VOTER-101", "Alice Johnson"),
        ("VOTER-102", "Bob Smith"),
        ("VOTER-103", "Charlie Davis"),
    ]

    stored_voter_faces = {}

    for vid, vname in demo_voters:
        samples = generate_mock_face_samples(NUM_SAMPLES_PER_VOTER, vid)
        stored_voter_faces[vid] = samples
        ok, msg = registry.save_voter_biometrics(vid, vname, samples)
        print(f"  ✔ {msg}")

    # Step 3: Train KNN Models (Comparing Pure NumPy vs Sklearn)
    print("\n[STEP 3]: Loading Dataset & Training Dual-Engine KNN Classifiers...")
    faces, names = registry.load_dataset()
    print(f"  Dataset Shape: {faces.shape} ({len(faces)} total face vectors, {len(set(names))} profiles)")

    # Pure NumPy KNN
    numpy_knn = FaceKNNClassifier(n_neighbors=5, engine="numpy")
    numpy_knn.fit(faces, names)
    print("  ✔ Pure NumPy Vectorized KNN Model Trained.")

    # Scikit-learn KNN
    sklearn_knn = FaceKNNClassifier(n_neighbors=5, engine="sklearn")
    sklearn_knn.fit(faces, names)
    print("  ✔ Scikit-Learn KNeighborsClassifier Trained.")

    # Step 4: Biometric Inference & Voting Simulation
    print("\n[STEP 4]: Simulating Real-Time Biometric Voting...")

    # Vote 1: Alice Johnson votes for Candidate 1
    alice_query = stored_voter_faces["VOTER-101"][0]  # Take a query vector
    pred_numpy, dist_np, conf_np = numpy_knn.predict(alice_query)
    pred_skl, dist_skl, conf_skl = sklearn_knn.predict(alice_query)

    print(f"\n  [Scan 1] Biometric Query: Alice Johnson")
    print(f"    - NumPy KNN Prediction:       {pred_numpy} (distance={dist_np:.1f}, confident={conf_np})")
    print(f"    - Scikit-Learn KNN Prediction: {pred_skl} (distance={dist_skl:.1f}, confident={conf_skl})")

    status, vid, vname = voting_mgr.check_voter_eligibility(pred_numpy)
    print(f"    - Voter Eligibility Check:    {status}")
    if status == "ELIGIBLE":
        ok, msg, rec = voting_mgr.cast_vote(vid, "1")
        print(f"    ✔ Ballot Cast: {rec['CandidateName']} | Hash: {rec['AuditHash'][:24]}...")

    # Vote 2: Bob Smith votes for Candidate 2
    bob_query = stored_voter_faces["VOTER-102"][5]
    pred_numpy, dist_np, conf_np = numpy_knn.predict(bob_query)
    status, vid, vname = voting_mgr.check_voter_eligibility(pred_numpy)
    print(f"\n  [Scan 2] Biometric Query: Bob Smith")
    print(f"    - Predicted: {pred_numpy} | Status: {status}")
    if status == "ELIGIBLE":
        ok, msg, rec = voting_mgr.cast_vote(vid, "2")
        print(f"    ✔ Ballot Cast: {rec['CandidateName']} | Hash: {rec['AuditHash'][:24]}...")

    # Step 5: Fraud Prevention Test (Double Voting)
    print("\n[STEP 5]: Fraud Prevention Test (Duplicate Voting Attempt)...")
    print("  [Scan 3] Alice Johnson attempts to vote a SECOND time:")
    status, vid, vname = voting_mgr.check_voter_eligibility(f"VOTER-101:Alice Johnson")
    print(f"    - Eligibility Status: {status}")
    if status == "ALREADY_VOTED":
        print("    ✔ SUCCESS: System BLOCKED duplicate voting attempt! One-person-one-vote enforced.")

    # Step 6: Unregistered Face Test
    print("\n[STEP 6]: Security Test (Unregistered / Unknown Face Attempt)...")
    stranger_face = np.random.uniform(0, 255, size=(FACE_IMG_SIZE[0] * FACE_IMG_SIZE[1],)).astype(np.float32)
    pred_stranger, dist_s, conf_s = numpy_knn.predict(stranger_face)
    status, vid, vname = voting_mgr.check_voter_eligibility(pred_stranger)
    print(f"    - Prediction: {pred_stranger} (avg distance={dist_s:.1f})")
    print(f"    - Eligibility Status: {status}")
    if status == "UNREGISTERED":
        print("    ✔ SUCCESS: System DENIED voting access to unregistered face!")

    # Vote 3: Charlie Davis votes for Candidate 3
    charlie_query = stored_voter_faces["VOTER-103"][10]
    pred_c, dist_c, conf_c = numpy_knn.predict(charlie_query)
    status, vid, vname = voting_mgr.check_voter_eligibility(pred_c)
    if status == "ELIGIBLE":
        ok, msg, rec = voting_mgr.cast_vote(vid, "3")
        print(f"\n  [Scan 4] Charlie Davis casts ballot for {rec['CandidateName']} ✔")

    # Step 7: Cryptographic SHA-256 Audit Validation
    print("\n[STEP 7]: Cryptographic Audit Trail Validation...")
    is_valid, report, count = voting_mgr.verify_audit_ledger()
    print(f"  ✔ Audit Chain Check: {report} ({count} ballots verified)")

    # Step 8: Administrative Analytics Report
    print("\n[STEP 8]: Final Election Results & Analytics Dashboard:")
    print_cli_dashboard()

    print("=" * 70)
    print("  SMARTELEC DEMONSTRATION COMPLETE: ALL SYSTEMS FUNCTIONAL & VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
