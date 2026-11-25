"""Summarize a rendered-scene window from Victory's frametime CSV."""
import argparse
import csv
import json
from pathlib import Path


def summarize(rows, start, end=None, excluded=()):
    selected = []
    for row in rows:
        last = int(row["frame"])
        count = int(float(row["frames"]))
        first = last - count + 1
        if count <= 0 or first < start or (end is not None and last > end):
            continue
        if any(first <= frame <= last for frame in excluded):
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No complete summary intervals in the selected frame window")
    frames = sum(float(r["frames"]) for r in selected)
    duration = sum(float(r["frames"]) * float(r["ms_per_frame"]) for r in selected) / 1000
    if duration <= 0:
        raise ValueError("Selected intervals have no elapsed time")
    def average(key):
        return sum(float(r["frames"]) * float(r[key]) for r in selected) / frames
    costs = ["cpu_prepare_ms", "gpu_ms", "copyback_ms", "copies_ms", "textures_ms", "heap_ms"]
    return {
        "first_frame": int(selected[0]["frame"]) - int(float(selected[0]["frames"])) + 1,
        "last_frame": int(selected[-1]["frame"]),
        "intervals": len(selected),
        "frames": int(frames),
        "seconds": round(duration, 3),
        "average_fps": round(frames / duration, 3),
        "average_frame_ms": round(duration * 1000 / frames, 3),
        "worst_frame_ms": max(float(r["frame_ms_max"]) for r in selected),
        "draws_per_frame": round(average("draws"), 1),
        "dropped_per_frame": round(average("dropped"), 1),
        "overlapping_costs_ms": {k: round(average(k), 3) for k in costs},
        "mean_working_set_mb": round(average("ram_mb"), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start-frame", type=int, required=True,
                        help="First rendered-scene frame to include after warmup")
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--exclude-frame", type=int, action="append", default=[],
                        help="Exclude the whole interval containing a capture; repeat as needed")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        try:
            result = summarize(rows, args.start_frame, args.end_frame, args.exclude_frame)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2))
