"""
SmartElec Configuration Module
Central repository for system constants, directory paths, camera configurations,
candidate registries, and ML hyperparameters.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Data storage paths
FACES_DATA_PATH = DATA_DIR / "faces_data.pkl"
NAMES_DATA_PATH = DATA_DIR / "names.pkl"
VOTERS_CSV_PATH = DATA_DIR / "voters.csv"
VOTES_CSV_PATH = DATA_DIR / "votes.csv"

# Computer Vision & Preprocessing Settings
FACE_IMG_SIZE = (50, 50)          # Dimensions for flattened feature vector (50 * 50 = 2500 features)
NUM_SAMPLES_PER_VOTER = 50        # Number of face snapshots captured during registration
SAMPLE_CAPTURE_DELAY = 0.05       # Delay between captures in seconds
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# Machine Learning Hyperparameters
KNN_K_NEIGHBORS = 5               # Value of 'k' for K-Nearest Neighbors
KNN_METRIC = "euclidean"          # Metric for distance computation
UNKNOWN_DISTANCE_THRESHOLD = 4500 # Euclidean distance threshold to reject unregistered faces

# Election Candidates
CANDIDATES = {
    "1": {"id": "CAN-01", "name": "Elena Rostova", "party": "Tech Innovation Alliance (TIA)", "symbol": "🚀"},
    "2": {"id": "CAN-02", "name": "Marcus Vance", "party": "Green Future Coalition (GFC)", "symbol": "🌱"},
    "3": {"id": "CAN-03", "name": "Priya Sharma", "party": "United Democratic League (UDL)", "symbol": "⚖️"},
    "4": {"id": "CAN-04", "name": "NOTA", "party": "None of the Above", "symbol": "❌"},
}

# CSV Headers
VOTERS_CSV_HEADERS = [
    "VoterID", "FullName", "RegistrationDate", "FaceVectorLength", "Status"
]

VOTES_CSV_HEADERS = [
    "VoteID", "VoterID", "VoterName", "CandidateID", "CandidateName", "Party", "Timestamp", "AuditHash"
]
