import krpc

class KRPCContext:
    def __init__(self):
        self.conn = krpc.connect(name="rover-agent")
        self.sc = self.conn.space_center
        self.vessel = self.sc.active_vessel
        self.ctrl = self.vessel.control
        self.body = self.vessel.orbit.body

    def close(self):
        try:
            self.ctrl.wheel_throttle = 0.0
            self.ctrl.wheel_steering = 0.0
            self.ctrl.brakes = False
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass
