"""
SmartElec Voting Manager Module
Enforces election integrity rules: duplicate-vote prevention, ballot recording,
real-time vote aggregation, and cryptographic SHA-256 audit-chain verification.
"""

import os
import csv
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .config import (
    DATA_DIR,
    VOTES_CSV_PATH,
    VOTES_CSV_HEADERS,
    CANDIDATES
)
from .voter_registry import VoterRegistry


class VotingManager:
    """
    Coordinates ballot casting, double-voting prevention, and election audit trails.
    """

    def __init__(self, registry: Optional[VoterRegistry] = None):
        self.registry = registry or VoterRegistry()
        self._ensure_votes_csv_initialized()

    def _ensure_votes_csv_initialized(self) -> None:
        """Initializes votes.csv with headers if absent."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not VOTES_CSV_PATH.exists():
            with open(VOTES_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(VOTES_CSV_HEADERS)

    def has_voter_already_voted(self, voter_id: str) -> bool:
        """
        Fraud Prevention Check:
        Scans votes.csv to verify whether the voter has already cast a ballot.
        """
        if not VOTES_CSV_PATH.exists():
            return False

        target_id = voter_id.strip().upper()
        with open(VOTES_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("VoterID", "").strip().upper() == target_id:
                    return True
        return False

    def check_voter_eligibility(self, voter_tag: str) -> Tuple[str, str, str]:
        """
        Evaluates voter eligibility based on face recognition label tag.
        Label tag format: 'VOTER_ID:Voter Name' or 'Unknown' / 'Unregistered'

        Returns:
            Tuple of (status_code, voter_id, voter_name)
            status_code can be:
                - 'ELIGIBLE': Voter is registered and has not voted yet
                - 'ALREADY_VOTED': Voter has already cast a ballot
                - 'UNREGISTERED': Face does not match any registered profile
        """
        if not voter_tag or voter_tag in ["Unknown", "Unregistered"]:
            return "UNREGISTERED", "", "Unregistered"

        if ":" in voter_tag:
            parts = voter_tag.split(":", 1)
            voter_id = parts[0].strip().upper()
            voter_name = parts[1].strip()
        else:
            voter_id = voter_tag.strip().upper()
            voter_name = voter_tag.strip()

        # Check if actually exists in registry
        if not self.registry.is_voter_id_registered(voter_id):
            return "UNREGISTERED", voter_id, voter_name

        # Check duplicate voting
        if self.has_voter_already_voted(voter_id):
            return "ALREADY_VOTED", voter_id, voter_name

        return "ELIGIBLE", voter_id, voter_name

    def _get_last_audit_hash(self) -> str:
        """Retrieves the cryptographic hash of the most recent vote for chaining."""
        if not VOTES_CSV_PATH.exists():
            return "GENESIS_HASH_SMARTELEC_0000000000000000"

        last_hash = "GENESIS_HASH_SMARTELEC_0000000000000000"
        with open(VOTES_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audit_hash = row.get("AuditHash")
                if audit_hash:
                    last_hash = audit_hash
        return last_hash

    def _generate_audit_hash(self, prev_hash: str, vote_id: str, voter_id: str, candidate_id: str, timestamp: str) -> str:
        """
        Computes SHA-256 tamper-evident hash chaining for election integrity.
        Formula: SHA256(prev_hash + vote_id + voter_id + candidate_id + timestamp)
        """
        payload = f"{prev_hash}|{vote_id}|{voter_id}|{candidate_id}|{timestamp}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def cast_vote(self, voter_id: str, candidate_key: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Executes a secure ballot casting transaction.

        Args:
            voter_id: Unique voter identifier
            candidate_key: Selection key ('1', '2', '3', '4')

        Returns:
            Tuple of (success, message, vote_record_dict)
        """
        voter_id = voter_id.strip().upper()

        # 1. Eligibility re-validation
        voter_info = self.registry.get_voter_by_id(voter_id)
        if not voter_info:
            return False, f"Voter ID '{voter_id}' is not in the official voter registry.", None

        if self.has_voter_already_voted(voter_id):
            return False, f"SECURITY ALERT: Voter '{voter_info['FullName']}' ({voter_id}) has already cast a vote!", None

        # 2. Candidate validation
        candidate = CANDIDATES.get(str(candidate_key))
        if not candidate:
            return False, f"Invalid candidate selection '{candidate_key}'.", None

        # 3. Create cryptographic vote record
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        vote_id = f"VOTE-{int(now.timestamp() * 1000)}"
        prev_hash = self._get_last_audit_hash()
        audit_hash = self._generate_audit_hash(prev_hash, vote_id, voter_id, candidate["id"], timestamp)

        record = {
            "VoteID": vote_id,
            "VoterID": voter_id,
            "VoterName": voter_info["FullName"],
            "CandidateID": candidate["id"],
            "CandidateName": candidate["name"],
            "Party": candidate["party"],
            "Timestamp": timestamp,
            "AuditHash": audit_hash
        }

        # 4. Atomic append to votes.csv
        with open(VOTES_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["VoteID"],
                record["VoterID"],
                record["VoterName"],
                record["CandidateID"],
                record["CandidateName"],
                record["Party"],
                record["Timestamp"],
                record["AuditHash"]
            ])

        return True, f"Vote successfully recorded for {candidate['name']} ({candidate['party']}).", record

    def get_election_results(self) -> Dict[str, Any]:
        """
        Aggregates election votes, calculates turnout percentages, and checks winner.
        """
        total_registered = len(self.registry.get_all_voters())
        candidate_tallies = {c["id"]: {"name": c["name"], "party": c["party"], "votes": 0, "symbol": c.get("symbol", "")} for c in CANDIDATES.values()}

        votes_cast = 0
        vote_log = []

        if VOTES_CSV_PATH.exists():
            with open(VOTES_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    votes_cast += 1
                    cid = row.get("CandidateID")
                    if cid in candidate_tallies:
                        candidate_tallies[cid]["votes"] += 1
                    vote_log.append(row)

        turnout_percent = (votes_cast / total_registered * 100.0) if total_registered > 0 else 0.0

        # Determine winner
        leading_candidate = None
        max_votes = -1
        for cid, data in candidate_tallies.items():
            if data["votes"] > max_votes:
                max_votes = data["votes"]
                leading_candidate = data

        return {
            "total_registered": total_registered,
            "total_votes_cast": votes_cast,
            "turnout_percentage": round(turnout_percent, 2),
            "candidate_tallies": candidate_tallies,
            "leading_candidate": leading_candidate if votes_cast > 0 else None,
            "vote_records": vote_log
        }

    def verify_audit_ledger(self) -> Tuple[bool, str, int]:
        """
        Cryptographically validates the entire SHA-256 chain in votes.csv to detect any tampering.
        Returns:
            Tuple of (is_valid, report_message, checked_records_count)
        """
        if not VOTES_CSV_PATH.exists():
            return True, "No votes cast yet. Audit ledger is clean.", 0

        prev_hash = "GENESIS_HASH_SMARTELEC_0000000000000000"
        records_checked = 0

        with open(VOTES_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                records_checked += 1
                expected_hash = self._generate_audit_hash(
                    prev_hash,
                    row["VoteID"],
                    row["VoterID"],
                    row["CandidateID"],
                    row["Timestamp"]
                )
                actual_hash = row["AuditHash"]
                if expected_hash != actual_hash:
                    return False, f"TAMPERING DETECTED at record {idx} (VoteID: {row['VoteID']})! Hash mismatch.", records_checked
                prev_hash = actual_hash

        return True, f"Integrity Verified: All {records_checked} vote audit hashes are authentic and uncompromised.", records_checked

    def reset_election(self) -> None:
        """Clears votes.csv for a new election."""
        if VOTES_CSV_PATH.exists():
            os.remove(VOTES_CSV_PATH)
        self._ensure_votes_csv_initialized()
