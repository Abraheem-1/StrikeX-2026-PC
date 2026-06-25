#!/usr/bin/env python3
"""
StrikeX pitch BREAKAWAY mapper (for a friction-locked axis).

The pitch axis self-locks by friction (it holds with power off), so there is no
gravity hold force to measure — that's why the old hold-test read 0 everywhere.
What actually matters for MOTION is the breakaway PWM: the push needed to start
the barrel moving from a standstill, against static friction.

At each angle this ramps PWM up from 0 until the encoder first moves, records
that breakaway value, then does the same in the other direction. Output:
    angle , breakaway_up , breakaway_down
which is exactly what the new stiction-based controller needs.

Requires firmware v1.14+ (raw pitch test drive via /tuning/pitch_test).
Keep the E-STOP in hand. Ctrl-C aborts and turns the motor off.
"""

import time
import roslibpy

PI_IP = "192.168.137.7"
ROSBRIDGE_PORT = 9090

ANGLE_START   = -30.0     # deg
ANGLE_END     =  50.0     # deg (stay inside the +58 limit)
ANGLE_STEP    =  10.0     # deg between measurements (10 = fast; lower = finer)
ANGLE_TOL     =   3.0     # deg: "close enough" when positioning

PWM_RAMP_START =   5      # begin ramp here
PWM_RAMP_MAX   = 200      # give up if it hasn't moved by here
PWM_RAMP_STEP  =   3      # increase by this each tick
RAMP_TICK_S    =  0.12    # wait between increments (let it react)
MOVE_THRESH    =   1.0    # deg of encoder change = "it moved"
SETTLE_S       =   0.6    # pause between sub-tests


class BreakawayMapper:
    def __init__(self):
        self.client = roslibpy.Ros(host=PI_IP, port=ROSBRIDGE_PORT)
        self.client.run()
        self.pub = roslibpy.Topic(self.client, '/tuning/pitch_test',
                                  'std_msgs/Float32MultiArray')
        self.elev_topic = roslibpy.Topic(self.client, '/gun/elevation_angle',
                                         'std_msgs/Float32')
        self.elev = None
        self.elev_topic.subscribe(self._on_elev)
        time.sleep(1.0)
        print(f"[mapper] connected; elevation = {self.elev}")

    def _on_elev(self, msg):
        self.elev = float(msg['data'])

    def set_pwm(self, pwm):
        self.pub.publish(roslibpy.Message({'data': [float(int(pwm)), 0.0]}))

    def off(self):
        self.pub.publish(roslibpy.Message({'data': [0.0, 1.0]}))

    def goto_angle(self, target):
        print(f"[mapper] positioning to {target:+.1f} deg ...")
        for _ in range(500):
            if self.elev is None:
                time.sleep(0.1); continue
            err = target - self.elev
            if abs(err) <= ANGLE_TOL:
                self.set_pwm(0); time.sleep(0.3); return True
            # ramp to break free, then nudge — direction of error
            push = 90 if err > 0 else -90
            self.set_pwm(push)
            time.sleep(0.12)
            self.set_pwm(0)          # pulse so it doesn't run away
            time.sleep(0.05)
        self.set_pwm(0)
        return abs(target - (self.elev or 0)) <= ANGLE_TOL

    # ramp PWM in `direction` (+1 up / -1 down) until the barrel moves
    def find_breakaway(self, direction):
        a0 = self.elev
        pwm = PWM_RAMP_START
        while pwm <= PWM_RAMP_MAX:
            self.set_pwm(direction * pwm)
            time.sleep(RAMP_TICK_S)
            if self.elev is not None and abs(self.elev - a0) >= MOVE_THRESH:
                self.set_pwm(0)
                return pwm                      # broke free here
            pwm += PWM_RAMP_STEP
        self.set_pwm(0)
        return None                             # never moved within PWM_RAMP_MAX

    def run(self):
        results = []
        ang = ANGLE_START
        try:
            while ang <= ANGLE_END + 0.01:
                if not self.goto_angle(ang):
                    print(f"[mapper] couldn't reach {ang:+.1f}, skipping")
                    ang += ANGLE_STEP; continue
                time.sleep(SETTLE_S)
                here = self.elev
                up = self.find_breakaway(+1)
                time.sleep(SETTLE_S)
                # reposition if the up-test moved it far
                if self.elev is not None and abs(self.elev - here) > ANGLE_STEP:
                    self.goto_angle(ang); time.sleep(SETTLE_S)
                down = self.find_breakaway(-1)
                time.sleep(SETTLE_S)
                us = f"{up}" if up is not None else "none"
                ds = f"{down}" if down is not None else "none"
                print(f"[mapper] >>> {here:+.1f} deg   UP={us}  DOWN={ds}\n")
                results.append((here, up, down))
                ang += ANGLE_STEP
        finally:
            self.off()

        print("\n===============  BREAKAWAY MAP  ===============")
        print(" angle_deg , break_up , break_down")
        for a, u, d in results:
            print(f" {a:+7.1f} , {u if u is not None else 'none':>5} , {d if d is not None else 'none':>5}")
        print("==============================================")
        print("break_up/down = PWM needed to start moving up/down at that angle.")
        print("Paste this whole table back.")
        self.client.terminate()


if __name__ == "__main__":
    print("StrikeX BREAKAWAY mapper — keep the E-STOP in hand.")
    print("It ramps PWM until the barrel starts moving, at each angle. Ctrl-C aborts.\n")
    input("Press ENTER when ready (barrel free, e-stop ready)...")
    BreakawayMapper().run()
