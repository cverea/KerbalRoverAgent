import csv
import time

class Logger:
    def __init__(self, filename="rover_log.csv"):
        self.filename = filename
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)

        # header
        self.writer.writerow([
            "step",
            "timestamp",
            "distance_m",
            "forward_m",
            "right_m",
            "speed_mps",
            "bearing_deg",
            "tool",
            "throttle",
            "steer",
            "brake_strength",
            "duration_s",
            "overridden_by_safety"
        ])
        self.file.flush()

    def log(self, step, rover_state, tool_name, args, overridden=False):
        self.writer.writerow([
            step,
            round(time.time(), 2),
            round(rover_state.get("distance_meters", 0), 2),
            round(rover_state.get("forward_distance_meters", 0), 2),
            round(rover_state.get("right_distance_meters", 0), 2),
            round(rover_state.get("speed_mps", 0), 2),
            round(rover_state.get("bearing_error_deg", 0), 2),
            tool_name,
            round(args.get("throttle", 0), 2),
            round(args.get("steer", 0), 2),
            round(args.get("brake_strength", 0), 2),
            round(args.get("duration_s", 0), 2),
            overridden
        ])
        self.file.flush()

    def close(self):
        self.file.close()