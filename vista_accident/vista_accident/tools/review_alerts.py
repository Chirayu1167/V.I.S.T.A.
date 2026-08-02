#!/usr/bin/env python3
"""
Alert review / feedback loop.

Every confirmed alert is already logged with its full `meta` (iou, speeds,
drop ratios, etc.) in alerts.jsonl. This tool lets you walk through them
and label each as a true/false positive, producing a small labeled dataset
you can use to actually justify a threshold change instead of guessing —
e.g. "these 6 false-positive anomaly_stops all had duration between 2.0
and 2.4s at an intersection we hadn't drawn a stop_zone for" is a real
tuning signal; "let's try raising anomaly_stop_duration_s" isn't, on its own.

Usage:
    python -m vista_accident.tools.review_alerts --log alerts.jsonl
    # walks through each alert not yet reviewed:
    #   [t/f/s/q] true positive / false positive / skip / quit
    # writes labels to alerts_review.jsonl (--review-out)

    python -m vista_accident.tools.review_alerts --log alerts.jsonl --summary
    # just prints false-positive-rate-by-kind from existing labels, no prompts
"""

import argparse
import json
import os
from collections import defaultdict


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _reviewed_ids(review_path):
    return {r["alert_id"] for r in _load_jsonl(review_path)}


def print_summary(review_path):
    labels = _load_jsonl(review_path)
    if not labels:
        print("No labeled alerts yet.")
        return
    by_kind = defaultdict(lambda: {"tp": 0, "fp": 0})
    for r in labels:
        by_kind[r["kind"]]["tp" if r["label"] == "true_positive" else "fp"] += 1

    print(f"{'kind':<15} {'tp':>4} {'fp':>4} {'fp_rate':>8}")
    for kind, counts in sorted(by_kind.items()):
        total = counts["tp"] + counts["fp"]
        rate = counts["fp"] / total if total else 0.0
        print(f"{kind:<15} {counts['tp']:>4} {counts['fp']:>4} {rate:>7.0%}")


def review_interactive(log_path, review_path):
    alerts = _load_jsonl(log_path)
    already = _reviewed_ids(review_path)
    pending = [a for a in alerts if a.get("alert_id") not in already]

    if not pending:
        print("Nothing new to review.")
        return

    print(f"{len(pending)} alert(s) to review. [t]rue positive  [f]alse positive  [s]kip  [q]uit\n")
    with open(review_path, "a") as out:
        for a in pending:
            print("-" * 60)
            print(f"alert_id={a.get('alert_id')}  kind={a.get('kind')}  severity={a.get('severity')}")
            print(f"  camera={a.get('camera_id')}  t={a.get('timestamp')}  tracks={a.get('track_ids')}")
            print(f"  meta={a.get('meta')}")
            choice = input("  label [t/f/s/q]: ").strip().lower()
            if choice == "q":
                break
            if choice not in ("t", "f"):
                continue
            label = "true_positive" if choice == "t" else "false_positive"
            record = {"alert_id": a.get("alert_id"), "kind": a.get("kind"),
                      "severity": a.get("severity"), "label": label, "meta": a.get("meta")}
            out.write(json.dumps(record) + "\n")
            out.flush()
    print("\nSaved. Run with --summary to see false-positive rate by kind.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="alerts.jsonl", help="Alerts log to review")
    ap.add_argument("--review-out", default="alerts_review.jsonl", help="Where labels are appended")
    ap.add_argument("--summary", action="store_true", help="Print false-positive rate by kind and exit")
    args = ap.parse_args()

    if args.summary:
        print_summary(args.review_out)
    else:
        review_interactive(args.log, args.review_out)


if __name__ == "__main__":
    main()
