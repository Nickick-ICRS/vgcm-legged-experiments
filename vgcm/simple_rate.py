import time

class Rate:
    def __init__(self, fps):
        self.fps = fps
        self.frame_duration = 1. / fps
        self.last_time = time.time()

    def sleep(self):
        now = time.time()
        dt = now - self.last_time
        sleep_dur = self.frame_duration - dt
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        self.last_time += sleep_dur
