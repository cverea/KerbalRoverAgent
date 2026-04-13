import time
import math

import wheel

class Executor:
    def __init__(self, ctrl, vessel, target, sc, state_tracker,
                 sleep_interval=0.05, max_duration_s=5.0):
        self.ctrl = ctrl
        self.vessel = vessel
        self.target = target
        self.sc = sc
        self.state_tracker = state_tracker
        self.sleep_interval = float(sleep_interval)
        self.max_duration_s = float(max_duration_s)


    # Sanitize and clamp action values to valid ranges, and handle NaNs by treating them as zero with brakes on
    def _sanitize_action(self, action):
        th = float(action.get("throttle", 0.0)) #
        st = float(action.get("steering", 0.0))
        br_strength = float(action.get("brake_strength", 0.0))
        br = bool(action.get("brakes", False))


        # clamp and handle NaN
        if math.isnan(th) or math.isnan(st) or math.isnan(br_strength):
            th, st, br_strength, br = 0.0, 0.0, 0.0, True

        th = max(-1.0, min(1.0, th))   
        st = max(-1.0, min(1.0, st))
        br_strength = max(0.0, min(100.0, br_strength))
        return {"throttle": th, "steering": st, "brake_strength": br_strength, "brakes": br}

    # Apply the action to the rover's controls
    def apply_action(self, action):
        a = self._sanitize_action(action)
        self.ctrl.wheel_throttle = a["throttle"]
        self.ctrl.wheel_steering = -a["steering"]        
        self.set_brake_strength(a["brake_strength"])
        self.ctrl.brakes = a["brakes"]

    def drive(self, throttle, steer, duration_s):
        action = {"throttle": throttle, "steering": steer, "brake_strength": 0.0, "brakes": False}
        return self.execute_action(action, duration_s)

    def brake(self, brake_strength, duration_s):
        action = {"throttle": 0.0, "steering": 0.0, "brake_strength": brake_strength, "brakes": True}
        return self.execute_action(action, duration_s)
    

    # Execute the action for the specified duration, with a fail-safe stop at the end
    def execute_action(self, action, duration_s):
        try:
            # validate duration
            duration_s = float(duration_s)
            if math.isnan(duration_s):
                duration_s = 0.0
            duration_s = max(0.0, min(self.max_duration_s, duration_s))

            self.apply_action(action)

            # simple timing loop to wait for the duration while allowing for interruption
            t_end = time.monotonic() + duration_s
            while time.monotonic() < t_end:
                time.sleep(self.sleep_interval)

        finally:
            self.apply_action({"throttle": 0.0, "steering": 0.0, "brake_strength": 0.0,"brakes": False})       
        
        return "OK"
    
    def set_brake_strength(self, brake_strength):
        for wheel in self.vessel.parts.wheels:
            wheel.brakes = brake_strength
