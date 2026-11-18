"""
SmartElec - Administrative Audit & Election Analytics Dashboard
Provides election metrics, candidate tallies, turnout percentages,
cryptographic SHA-256 audit-chain validation, and GUI/CLI interfaces.
"""

import sys
import argparse
import csv
from datetime import datetime
from typing import Dict, Any

from smartelec.voting_manager import VotingManager
from smartelec.voter_registry import VoterRegistry
from smartelec.config import CANDIDATES, VOTES_CSV_PATH, VOTERS_CSV_PATH

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


def print_cli_dashboard():
    """Renders a formatted executive election summary in the console."""
    registry = VoterRegistry()
    voting_mgr = VotingManager(registry=registry)
    results = voting_mgr.get_election_results()
    is_valid, audit_msg, count = voting_mgr.verify_audit_ledger()

    print("\n" + "=" * 70)
    print("        SMARTELEC - ELECTION ADMINISTRATION & AUDIT REPORT")
    print("=" * 70)
    print(f" Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Registered Voters:    {results['total_registered']}")
    print(f" Total Votes Cast:     {results['total_votes_cast']}")
    print(f" Voter Turnout:        {results['turnout_percentage']}%")

    # Winner badge
    leading = results["leading_candidate"]
    if leading and results["total_votes_cast"] > 0:
        print(f" Current Leader:       {leading['name']} ({leading['party']}) with {leading['votes']} votes")
    else:
        print(f" Current Leader:       No votes recorded yet.")
    print("-" * 70)

    # Candidate Breakdown Table
    print("\n[CANDIDATE RESULTS]")
    table_data = []
    for cid, data in results["candidate_tallies"].items():
        pct = (data["votes"] / results["total_votes_cast"] * 100) if results["total_votes_cast"] > 0 else 0.0
        table_data.append([
            cid,
            f"{data.get('symbol', '')} {data['name']}",
            data["party"],
            data["votes"],
            f"{pct:.1f}%"
        ])

    headers = ["ID", "Candidate Name", "Political Party", "Votes", "Vote %"]
    if TABULATE_AVAILABLE:
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    else:
        print(f"{'ID':<8} | {'Candidate Name':<25} | {'Party':<30} | {'Votes':<6} | {'Vote %':<7}")
        print("-" * 80)
        for row in table_data:
            print(f"{row[0]:<8} | {row[1]:<25} | {row[2]:<30} | {row[3]:<6} | {row[4]:<7}")

    # Cryptographic Audit Verification
    print("\n" + "-" * 70)
    print("[CRYPTOGRAPHIC AUDIT CHAIN VERIFICATION]")
    status_icon = "✔ SUCCESS" if is_valid else "✖ ALERT - CORRUPTED"
    print(f" Ledger Status:   {status_icon}")
    print(f" Audit Detail:    {audit_msg}")
    print(f" Records Audited: {count}")
    print("-" * 70)

    # Recent Audit Log Entries
    if results["vote_records"]:
        print("\n[RECENT AUDIT TRAIL LOGS (Last 5 Votes)]")
        log_rows = []
        for r in results["vote_records"][-5:]:
            log_rows.append([
                r.get("VoteID", "")[:14],
                r.get("VoterID", ""),
                r.get("VoterName", ""),
                r.get("CandidateName", "")[:18],
                r.get("Timestamp", "")[11:],
                r.get("AuditHash", "")[:16] + "..."
            ])
        log_headers = ["Vote ID", "Voter ID", "Voter Name", "Candidate", "Time", "SHA-256 Hash"]
        if TABULATE_AVAILABLE:
            print(tabulate(log_rows, headers=log_headers, tablefmt="simple"))
        else:
            for lr in log_rows:
                print(" | ".join(str(c) for c in lr))
    print("=" * 70 + "\n")


def launch_gui_dashboard():
    """Launches an interactive Tkinter GUI window with real-time refresh capability."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("[WARN] Tkinter is not available. Falling back to CLI report.")
        print_cli_dashboard()
        return

    registry = VoterRegistry()
    voting_mgr = VotingManager(registry=registry)

    root = tk.Tk()
    root.title("SmartElec - Administrative Audit Dashboard")
    root.geometry("900x680")
    root.configure(bg="#1e222b")

    # Styling
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background="#2a2f3b", foreground="#ffffff", fieldbackground="#2a2f3b", rowheight=26)
    style.map("Treeview", background=[("selected", "#0078d4")])
    style.configure("Treeview.Heading", background="#15181f", foreground="#00d2ff", font=("Segoe UI", 10, "bold"))

    # Header Banner
    header_frame = tk.Frame(root, bg="#15181f", pady=15, padx=20)
    header_frame.pack(fill=tk.X)

    title_lbl = tk.Label(header_frame, text="SmartElec - Election Administration & Audit Portal",
                         font=("Segoe UI", 16, "bold"), fg="#00d2ff", bg="#15181f")
    title_lbl.pack(anchor="w")

    subtitle_lbl = tk.Label(header_frame, text="Real-time vote tallies, turnout tracking & cryptographic SHA-256 audit ledger",
                            font=("Segoe UI", 10), fg="#a0aec0", bg="#15181f")
    subtitle_lbl.pack(anchor="w")

    # Metrics Summary Cards
    cards_frame = tk.Frame(root, bg="#1e222b", pady=15, padx=20)
    cards_frame.pack(fill=tk.X)

    def create_card(parent, title, val_var, color):
        frame = tk.Frame(parent, bg="#2a2f3b", padx=15, pady=12, highlightbackground=color, highlightthickness=1)
        lbl_t = tk.Label(frame, text=title.upper(), font=("Segoe UI", 8, "bold"), fg="#a0aec0", bg="#2a2f3b")
        lbl_t.pack(anchor="w")
        lbl_v = tk.Label(frame, textvariable=val_var, font=("Segoe UI", 18, "bold"), fg=color, bg="#2a2f3b")
        lbl_v.pack(anchor="w", pady=(5, 0))
        return frame

    var_reg = tk.StringVar(value="0")
    var_votes = tk.StringVar(value="0")
    var_turnout = tk.StringVar(value="0.0%")
    var_winner = tk.StringVar(value="None")
    var_audit = tk.StringVar(value="Verified ✔")

    card1 = create_card(cards_frame, "Registered Voters", var_reg, "#63b3ed")
    card2 = create_card(cards_frame, "Votes Cast", var_votes, "#68d391")
    card3 = create_card(cards_frame, "Voter Turnout", var_turnout, "#f6ad55")
    card4 = create_card(cards_frame, "Current Leader", var_winner, "#f687b3")
    card5 = create_card(cards_frame, "SHA-256 Audit", var_audit, "#4fd1c5")

    card1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
    card2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
    card3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
    card4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
    card5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

    # Notebook Tabs (Candidate Tallies, Audit Logs, Registered Voters)
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Tab 1: Candidate Tallies
    tab_tallies = tk.Frame(notebook, bg="#1e222b", padx=10, pady=10)
    notebook.add(tab_tallies, text="Candidate Standings")

    tree_tallies = ttk.Treeview(tab_tallies, columns=("ID", "Name", "Party", "Votes", "Percentage"), show="headings", height=6)
    tree_tallies.heading("ID", text="Candidate ID")
    tree_tallies.heading("Name", text="Candidate Name")
    tree_tallies.heading("Party", text="Party")
    tree_tallies.heading("Votes", text="Total Votes")
    tree_tallies.heading("Percentage", text="Vote Share (%)")

    tree_tallies.column("ID", width=110, anchor="center")
    tree_tallies.column("Name", width=220)
    tree_tallies.column("Party", width=260)
    tree_tallies.column("Votes", width=110, anchor="center")
    tree_tallies.column("Percentage", width=120, anchor="center")
    tree_tallies.pack(fill=tk.BOTH, expand=True)

    # Tab 2: Audit Ledger
    tab_audit = tk.Frame(notebook, bg="#1e222b", padx=10, pady=10)
    notebook.add(tab_audit, text="Live Audit Ledger (CSV)")

    tree_audit = ttk.Treeview(tab_audit, columns=("VoteID", "VoterID", "VoterName", "Candidate", "Timestamp", "AuditHash"), show="headings")
    tree_audit.heading("VoteID", text="Vote ID")
    tree_audit.heading("VoterID", text="Voter ID")
    tree_audit.heading("VoterName", text="Voter Name")
    tree_audit.heading("Candidate", text="Candidate")
    tree_audit.heading("Timestamp", text="Timestamp")
    tree_audit.heading("AuditHash", text="Cryptographic SHA-256 Hash")

    tree_audit.column("VoteID", width=120, anchor="center")
    tree_audit.column("VoterID", width=100, anchor="center")
    tree_audit.column("VoterName", width=150)
    tree_audit.column("Candidate", width=150)
    tree_audit.column("Timestamp", width=140, anchor="center")
    tree_audit.column("AuditHash", width=180)
    tree_audit.pack(fill=tk.BOTH, expand=True)

    # Tab 3: Registered Voters
    tab_voters = tk.Frame(notebook, bg="#1e222b", padx=10, pady=10)
    notebook.add(tab_voters, text="Registered Voters")

    tree_voters = ttk.Treeview(tab_voters, columns=("VoterID", "Name", "Date", "Status", "HasVoted"), show="headings")
    tree_voters.heading("VoterID", text="Voter ID")
    tree_voters.heading("Name", text="Full Name")
    tree_voters.heading("Date", text="Registration Date")
    tree_voters.heading("Status", text="Status")
    tree_voters.heading("HasVoted", text="Ballot Status")

    tree_voters.column("VoterID", width=110, anchor="center")
    tree_voters.column("Name", width=220)
    tree_voters.column("Date", width=160, anchor="center")
    tree_voters.column("Status", width=110, anchor="center")
    tree_voters.column("HasVoted", width=130, anchor="center")
    tree_voters.pack(fill=tk.BOTH, expand=True)

    def refresh_data():
        res = voting_mgr.get_election_results()
        var_reg.set(str(res["total_registered"]))
        var_votes.set(str(res["total_votes_cast"]))
        var_turnout.set(f"{res['turnout_percentage']}%")

        leader = res["leading_candidate"]
        if leader and res["total_votes_cast"] > 0:
            var_winner.set(f"{leader['name']} ({leader['votes']})")
        else:
            var_winner.set("No Votes")

        is_valid, msg, _ = voting_mgr.verify_audit_ledger()
        var_audit.set("Verified ✔" if is_valid else "CORRUPTED ✖")

        # Refresh Tallies Tree
        for row in tree_tallies.get_children():
            tree_tallies.delete(row)

        for cid, data in res["candidate_tallies"].items():
            pct = (data["votes"] / res["total_votes_cast"] * 100) if res["total_votes_cast"] > 0 else 0.0
            tree_tallies.insert("", tk.END, values=(
                cid,
                f"{data.get('symbol', '')} {data['name']}",
                data["party"],
                data["votes"],
                f"{pct:.1f}%"
            ))

        # Refresh Audit Tree
        for row in tree_audit.get_children():
            tree_audit.delete(row)

        for r in res["vote_records"]:
            tree_audit.insert("", tk.END, values=(
                r.get("VoteID", ""),
                r.get("VoterID", ""),
                r.get("VoterName", ""),
                r.get("CandidateName", ""),
                r.get("Timestamp", ""),
                r.get("AuditHash", "")
            ))

        # Refresh Voters Tree
        for row in tree_voters.get_children():
            tree_voters.delete(row)

        all_voters = registry.get_all_voters()
        for v in all_voters:
            voted = "VOTED ✔" if voting_mgr.has_voter_already_voted(v["VoterID"]) else "PENDING ⌛"
            tree_voters.insert("", tk.END, values=(
                v["VoterID"],
                v["FullName"],
                v["RegistrationDate"],
                v["Status"],
                voted
            ))

    # Control buttons at bottom
    btn_frame = tk.Frame(root, bg="#1e222b", pady=10, padx=20)
    btn_frame.pack(fill=tk.X)

    btn_refresh = tk.Button(btn_frame, text="🔄 Refresh Data", command=refresh_data,
                            font=("Segoe UI", 10, "bold"), bg="#0078d4", fg="white", padx=15, pady=5)
    btn_refresh.pack(side=tk.LEFT)

    def on_verify():
        is_val, msg, c = voting_mgr.verify_audit_ledger()
        if is_val:
            messagebox.showinfo("Audit Integrity Check", f"Integrity Verified!\n\nAll {c} vote records match their cryptographic SHA-256 signatures.")
        else:
            messagebox.showerror("Audit Alert", f"INTEGRITY VIOLATION!\n\n{msg}")

    btn_audit_check = tk.Button(btn_frame, text="🔒 Verify SHA-256 Ledger", command=on_verify,
                                font=("Segoe UI", 10), bg="#2d3748", fg="#00d2ff", padx=15, pady=5)
    btn_audit_check.pack(side=tk.LEFT, padx=10)

    refresh_data()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="SmartElec Administrative Audit Dashboard")
    parser.add_argument("--cli", action="store_true", help="Display audit summary directly in the command line terminal")
    args = parser.parse_args()

    if args.cli:
        print_cli_dashboard()
    else:
        # Check if GUI is supported; otherwise fallback to CLI
        try:
            launch_gui_dashboard()
        except Exception as e:
            print(f"[INFO] GUI launch failed ({e}). Showing CLI report instead:")
            print_cli_dashboard()


if __name__ == "__main__":
    main()
