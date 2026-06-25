#!/usr/bin/env python3
"""
StrikeX pitch gravity mapper.

For each target elevation angle, finds the PWM that holds the barrel STILL
(no drift up or down). That holding PWM is the gravity load at that angle,
measured directly. The result is a table: angle -> hold_pwm, which is exactly
what the feedforward should output. No model, no guessing.

Requires firmware v1.14+ which accepts a raw test command:
    {"pitch_test_pwm": <int>}   -> drives pitch at that raw PWM, bypassing the
                                   position loop / FF / bias (test mode)
    {"pitch_test_off": true}    -> exit test mode, motor off

Reads /gun/elevation_angle (deg) for the encoder.
Run with the barrel free to move and the e-stop within reach.
"""

import time
import roslibpy

PI_IP = "192.168.137.7"
ROSBRIDGE_PORT = 9090

# sweep settings
ANGLE_START   = -30.0     # deg (forward/down lean)
ANGLE_END     =  55.0     # deg (backward/up lean) — stay just inside the +58 limit
ANGLE_STEP    =   5.0     # measure every 5 deg
ANGLE_TOL     =   1.5     # deg: "at the angle" if within this
DRIFT_TOL     =   2.0     # deg/s: barrel counts as "held still" below this
SETTLE_S      =   0.8     # seconds to watch drift at each PWM trial
PWM_MIN       = -200      # search bounds (negative = down, positive = up)
PWM_MAX       =  200
PWM_RESOLUTION =   3      # stop binary search when bracket is this tight


class GravityMapper:
    def __init__(self):
        self.client = roslibpy.Ros(host=PI_IP, port=ROSBRIDGE_PORT)
        self.client.run()

        self.pitch_pub = roslibpy.Topic(
            self.client, '/tuning/pitch_test', 'std_msgs/Float32MultiArray'
        )

        self.elev_topic = roslibpy.Topic(
            self.client, '/gun/elevation_angle', 'std_msgs/Float32'
        )
        self.elev = None
        self.elev_topic.subscribe(self._on_elev)
        time.sleep(1.0)
        print(f"[mapper] connected; current elevation = {self.elev}")

    def _on_elev(self, msg):
        self.elev = float(msg['data'])

    def set_test_pwm(self, pwm):
        self.pitch_pub.publish(roslibpy.Message({'data': [float(int(pwm)), 0.0]}))

    def test_off(self):
        self.pitch_pub.publish(roslibpy.Message({'data': [0.0, 1.0]}))

    # measure drift rate (deg/s) over SETTLE_S while holding a PWM
    def measure_drift(self, pwm):
        self.set_test_pwm(pwm)
        time.sleep(0.25)                 # let it react
        a0 = self.elev
        t0 = time.time()
        time.sleep(SETTLE_S)
        a1 = self.elev
        dt = time.time() - t0
        if a0 is None or a1 is None or dt <= 0:
            return 0.0, a1
        return (a1 - a0) / dt, a1

    # drive the barrel to a target angle using a gentle bias, then stop
    def goto_angle(self, target):
        print(f"[mapper] moving to {target:+.1f} deg ...")
        for _ in range(400):             # ~ up to 40 s
            if self.elev is None:
                time.sleep(0.1); continue
            err = target - self.elev
            if abs(err) <= ANGLE_TOL:
                self.set_test_pwm(0)
                return True
            # crude approach push, proportional, clamped
            push = max(-120, min(120, int(err * 8)))
            self.set_test_pwm(push)
            time.sleep(0.1)
        return abs(target - (self.elev or 0)) <= ANGLE_TOL

    # binary-search the PWM that holds the barrel still at the CURRENT angle
    def find_hold_pwm(self):
        lo, hi = PWM_MIN, PWM_MAX
        # ensure bracket signs are opposite (lo drifts one way, hi the other)
        while hi - lo > PWM_RESOLUTION:
            mid = (lo + hi) // 2
            drift, ang = self.measure_drift(mid)
            sign = "up" if drift > 0 else ("down" if drift < 0 else "still")
            print(f"        pwm={mid:+4d}  drift={drift:+5.1f} deg/s ({sign})  ang={ang:+.1f}")
            if abs(drift) <= DRIFT_TOL:
                return mid, ang
            # positive drift = moving up -> need less up push -> lower hi
            if drift > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) // 2, self.elev

    def run(self):
        results = []
        ang = ANGLE_START
        try:
            while ang <= ANGLE_END + 0.01:
                if not self.goto_angle(ang):
                    print(f"[mapper] could not reach {ang:+.1f}, skipping")
                    ang += ANGLE_STEP
                    continue
                time.sleep(0.4)
                print(f"[mapper] at {self.elev:+.1f} deg — searching hold PWM")
                hold_pwm, at_ang = self.find_hold_pwm()
                print(f"[mapper] >>> {at_ang:+.1f} deg  HOLD PWM = {hold_pwm:+d}\n")
                results.append((at_ang, hold_pwm))
                ang += ANGLE_STEP
        finally:
            self.test_off()

        print("\n================  GRAVITY MAP  ================")
        print(" angle_deg , hold_pwm")
        for a, p in results:
            print(f" {a:+7.1f} , {p:+d}")
        print("==============================================")
        print("hold_pwm is the gravity load at each angle = what the")
        print("feedforward should output there. Paste this back.")

        # also dump as a C array for easy firmware paste
        print("\n// firmware paste:")
        print(f"const int GRAV_MAP_N = {len(results)};")
        angs = ", ".join(f"{a:.1f}f" for a, _ in results)
        pwms = ", ".join(f"{p}" for _, p in results)
        print(f"const float GRAV_MAP_DEG[GRAV_MAP_N] = {{ {angs} }};")
        print(f"const int   GRAV_MAP_PWM[GRAV_MAP_N] = {{ {pwms} }};")

        self.client.terminate()


if __name__ == "__main__":
    print("StrikeX gravity mapper — keep the E-STOP within reach.")
    print("This will drive the pitch motor through its range. Ctrl-C aborts.\n")
    input("Press ENTER when ready (barrel free, e-stop ready)...")
    GravityMapper().run()
