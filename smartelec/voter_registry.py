"""
SmartElec Voter Registry Module
Handles storage, serialization, validation, and retrieval of voter biometric data
and identity profiles across pickle files and CSV databases.
"""

import os
import pickle
import csv
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

from .config import (
    DATA_DIR,
    FACES_DATA_PATH,
    NAMES_DATA_PATH,
    VOTERS_CSV_PATH,
    VOTERS_CSV_HEADERS,
    FACE_IMG_SIZE
)


class VoterRegistry:
    """
    Manages persistent storage for voter biometric vectors and profile metadata.
    """

    def __init__(self):
        self._ensure_csv_initialized()

    def _ensure_csv_initialized(self) -> None:
        """Initializes voters.csv with headers if it does not already exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not VOTERS_CSV_PATH.exists():
            with open(VOTERS_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(VOTERS_CSV_HEADERS)

    def is_voter_id_registered(self, voter_id: str) -> bool:
        """Checks if a given Voter ID already exists in the registry."""
        if not VOTERS_CSV_PATH.exists():
            return False

        with open(VOTERS_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("VoterID", "").strip().upper() == voter_id.strip().upper():
                    return True
        return False

    def get_voter_by_id(self, voter_id: str) -> Optional[Dict[str, str]]:
        """Retrieves details of a registered voter by their ID."""
        if not VOTERS_CSV_PATH.exists():
            return None

        with open(VOTERS_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("VoterID", "").strip().upper() == voter_id.strip().upper():
                    return row
        return None

    def get_all_voters(self) -> List[Dict[str, str]]:
        """Returns all registered voters from voters.csv."""
        if not VOTERS_CSV_PATH.exists():
            return []

        voters = []
        with open(VOTERS_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                voters.append(row)
        return voters

    def save_voter_biometrics(
        self, voter_id: str, full_name: str, face_samples: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Appends new face feature vectors to faces_data.pkl, updates names.pkl,
        and logs profile metadata to voters.csv.

        Args:
            voter_id: Unique voter identifier (e.g., 'VOTER-101')
            full_name: Voter's full name
            face_samples: 2D array of shape (N, 2500) representing captured face vectors

        Returns:
            Tuple of (success_status, message)
        """
        voter_id = voter_id.strip().upper()
        full_name = full_name.strip()

        if not voter_id or not full_name:
            return False, "Voter ID and Full Name cannot be empty."

        if self.is_voter_id_registered(voter_id):
            return False, f"Voter ID '{voter_id}' is already registered in the system."

        face_samples = np.asarray(face_samples, dtype=np.float32)
        if face_samples.ndim != 2:
            return False, f"Expected 2D array of face vectors, got shape {face_samples.shape}."

        expected_dim = FACE_IMG_SIZE[0] * FACE_IMG_SIZE[1]
        if face_samples.shape[1] != expected_dim:
            return False, f"Face vector length must be {expected_dim}, got {face_samples.shape[1]}."

        num_samples = len(face_samples)
        # We store labels as "VoterID:FullName" for direct identification
        label_tag = f"{voter_id}:{full_name}"
        labels_to_add = [label_tag] * num_samples

        # 1. Update faces_data.pkl
        if FACES_DATA_PATH.exists():
            try:
                with open(FACES_DATA_PATH, "rb") as f:
                    existing_faces = pickle.load(f)
                updated_faces = np.vstack([existing_faces, face_samples])
            except Exception as e:
                return False, f"Failed reading existing face data: {e}"
        else:
            updated_faces = face_samples

        # 2. Update names.pkl
        if NAMES_DATA_PATH.exists():
            try:
                with open(NAMES_DATA_PATH, "rb") as f:
                    existing_names = pickle.load(f)
                if isinstance(existing_names, np.ndarray):
                    updated_names = np.concatenate([existing_names, np.array(labels_to_add)])
                else:
                    updated_names = existing_names + labels_to_add
            except Exception as e:
                return False, f"Failed reading existing labels data: {e}"
        else:
            updated_names = np.array(labels_to_add)

        # Atomic writes for pickles
        with open(FACES_DATA_PATH, "wb") as f:
            pickle.dump(updated_faces, f)

        with open(NAMES_DATA_PATH, "wb") as f:
            pickle.dump(updated_names, f)

        # 3. Append to voters.csv
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(VOTERS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                voter_id,
                full_name,
                timestamp,
                expected_dim,
                "REGISTERED"
            ])

        return True, f"Successfully registered voter '{full_name}' ({voter_id}) with {num_samples} samples."

    def load_dataset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Loads training feature vectors and target labels.
        Returns:
            Tuple of (X, y) or (None, None) if dataset is empty.
        """
        if not FACES_DATA_PATH.exists() or not NAMES_DATA_PATH.exists():
            return None, None

        try:
            with open(FACES_DATA_PATH, "rb") as f:
                faces = pickle.load(f)
            with open(NAMES_DATA_PATH, "rb") as f:
                names = pickle.load(f)

            faces = np.asarray(faces, dtype=np.float32)
            names = np.asarray(names)

            if len(faces) == 0 or len(names) == 0:
                return None, None

            return faces, names
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            return None, None

    def reset_registry(self) -> None:
        """Utility for administrative testing: removes registered face datasets."""
        for path in [FACES_DATA_PATH, NAMES_DATA_PATH, VOTERS_CSV_PATH]:
            if path.exists():
                os.remove(path)
        self._ensure_csv_initialized()
