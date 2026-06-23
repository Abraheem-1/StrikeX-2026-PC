import time
import threading

import matplotlib.pyplot as plt
import roslibpy


PI_IP = "192.168.137.7"
ROSBRIDGE_PORT = 9090
WINDOW_SECONDS = 20
HISTORY_LEN = 400


class LiveDebugPlot:

    def __init__(self):
        self.t_start = time.time()
        self.lock = threading.Lock()

        self.ex_data, self.ex_time = [], []
        self.ey_data, self.ey_time = [], []
        self.yaw_data, self.yaw_time = [], []
        self.pitch_data, self.pitch_time = [], []

        self._setup_plot()
        self._start_ros_listener()

    # ── Setup ──────────────────────────────────────────
    def _setup_plot(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 6))
        self.fig.suptitle('StrikeX Live PID Debug')

        self.line_ex, = self.ax1.plot([], [], label='error_x (ex)', color='tab:blue')
        self.line_yaw, = self.ax1.plot([], [], label='yaw_angle', color='tab:orange')
        self.ax1.set_ylabel('Yaw axis')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)

        self.line_ey, = self.ax2.plot([], [], label='error_y (ey)', color='tab:blue')
        self.line_pitch, = self.ax2.plot([], [], label='pitch_angle', color='tab:orange')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Pitch axis')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True)

        plt.tight_layout()
        plt.show(block=False)

    def _start_ros_listener(self):
        thread = threading.Thread(target=self._ros_loop, daemon=True)
        thread.start()

    def _ros_loop(self):
        try:
            client = roslibpy.Ros(host=PI_IP, port=ROSBRIDGE_PORT)
            client.run()

            yaw_topic = roslibpy.Topic(
                client, '/gun/current_direction', 'std_msgs/Float32'
            )
            pitch_topic = roslibpy.Topic(
                client, '/gun/elevation_angle', 'std_msgs/Float32'
            )

            yaw_topic.subscribe(self._on_yaw)
            pitch_topic.subscribe(self._on_pitch)

            print(f'[live_debug_plot] Connected to RosBridge at {PI_IP}:{ROSBRIDGE_PORT}')

            while True:
                time.sleep(1)
        except Exception as e:
            print(f'[live_debug_plot] ROS connection failed: {e}')

    def _on_yaw(self, msg):
        with self.lock:
            self.yaw_data.append(msg['data'])
            self.yaw_time.append(self._now())
            self._trim(self.yaw_data, self.yaw_time)

    def _on_pitch(self, msg):
        with self.lock:
            self.pitch_data.append(msg['data'])
            self.pitch_time.append(self._now())
            self._trim(self.pitch_data, self.pitch_time)

    # ── Public API ─────────────────────────────────────
    def record_error(self, error_x, error_y):
        if error_x is None or error_y is None:
            return
        with self.lock:
            t = self._now()
            self.ex_data.append(error_x)
            self.ex_time.append(t)
            self.ey_data.append(error_y)
            self.ey_time.append(t)
            self._trim(self.ex_data, self.ex_time)
            self._trim(self.ey_data, self.ey_time)

    def update(self):
        """Call once per main loop iteration — non-blocking redraw."""
        with self.lock:
            t_now = self._now()

            if self.ex_time:
                self.line_ex.set_data(self.ex_time, self.ex_data)
            if self.yaw_time:
                self.line_yaw.set_data(self.yaw_time, self.yaw_data)
            if self.ey_time:
                self.line_ey.set_data(self.ey_time, self.ey_data)
            if self.pitch_time:
                self.line_pitch.set_data(self.pitch_time, self.pitch_data)

        for ax in (self.ax1, self.ax2):
            ax.set_xlim(max(0, t_now - WINDOW_SECONDS), t_now + 1)
            ax.relim()
            ax.autoscale_view(scalex=False)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── Helpers ────────────────────────────────────────
    def _now(self):
        return time.time() - self.t_start

    def _trim(self, data_list, time_list):
        while len(data_list) > HISTORY_LEN:
            data_list.pop(0)
            time_list.pop(0)