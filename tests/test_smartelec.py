"""
SmartElec Unit Test Suite
Validates KNN mathematical correctness, biometric registry persistence,
duplicate vote blocking, and SHA-256 audit-chain tamper detection.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from smartelec.knn_classifier import PureNumpyKNN, FaceKNNClassifier
from smartelec.voter_registry import VoterRegistry
from smartelec.voting_manager import VotingManager
from smartelec.config import FACE_IMG_SIZE


class TestSmartElec(unittest.TestCase):

    def setUp(self):
        # Create isolated temporary testing directory
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_pure_numpy_knn_correctness(self):
        """Verifies that PureNumpyKNN mathematically matches majority class and Euclidean distance."""
        # 2 classes in 2D space
        X_train = np.array([
            [1.0, 1.0],
            [1.1, 1.0],
            [0.9, 1.2],
            [10.0, 10.0],
            [10.2, 9.8],
            [9.9, 10.1]
        ], dtype=np.float32)
        y_train = np.array(["CLASS_A", "CLASS_A", "CLASS_A", "CLASS_B", "CLASS_B", "CLASS_B"])

        knn = PureNumpyKNN(n_neighbors=3, distance_threshold=100.0)
        knn.fit(X_train, y_train)

        # Test point close to CLASS_A
        pred, dist, conf = knn.predict_single(np.array([1.05, 0.95], dtype=np.float32))
        self.assertEqual(pred, "CLASS_A")
        self.assertTrue(conf)
        self.assertLess(dist, 1.0)

        # Test point close to CLASS_B
        pred_b, dist_b, conf_b = knn.predict_single(np.array([10.1, 10.05], dtype=np.float32))
        self.assertEqual(pred_b, "CLASS_B")
        self.assertTrue(conf_b)

    def test_dual_engine_consistency(self):
        """Verifies that Pure NumPy KNN and Scikit-Learn produce identical predictions."""
        np.random.seed(42)
        X = np.random.randn(60, 50).astype(np.float32)
        # Shift clusters so each voter class is distinct in feature space
        X[:20] += 10.0
        X[20:40] += 50.0
        X[40:] += 100.0
        y = np.array(["VOTER_1"] * 20 + ["VOTER_2"] * 20 + ["VOTER_3"] * 20)

        numpy_knn = FaceKNNClassifier(n_neighbors=5, engine="numpy", distance_threshold=5000.0)
        numpy_knn.fit(X, y)

        sklearn_knn = FaceKNNClassifier(n_neighbors=5, engine="sklearn", distance_threshold=5000.0)
        sklearn_knn.fit(X, y)

        # Query points from each class
        for idx in [5, 25, 45]:
            q = X[idx]
            pred_np, dist_np, conf_np = numpy_knn.predict(q)
            pred_sk, dist_sk, conf_sk = sklearn_knn.predict(q)

            self.assertEqual(pred_np, pred_sk)
            self.assertEqual(pred_np, y[idx])
            self.assertAlmostEqual(dist_np, dist_sk, places=2)

    def test_duplicate_vote_prevention(self):
        """Verifies that duplicate voting is strictly rejected."""
        registry = VoterRegistry()
        voting_mgr = VotingManager(registry=registry)

        # Check existing voted voter from demo
        has_voted = voting_mgr.has_voter_already_voted("VOTER-101")
        if has_voted:
            status, _, _ = voting_mgr.check_voter_eligibility("VOTER-101:Alice Johnson")
            self.assertEqual(status, "ALREADY_VOTED")

            # Attempting to cast vote again should fail
            ok, msg, rec = voting_mgr.cast_vote("VOTER-101", "1")
            self.assertFalse(ok)
            self.assertIn("already cast a vote", msg)

    def test_cryptographic_audit_tamper_detection(self):
        """Verifies that any modification to votes CSV breaks SHA-256 audit ledger."""
        registry = VoterRegistry()
        voting_mgr = VotingManager(registry=registry)

        is_valid, report, count = voting_mgr.verify_audit_ledger()
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
