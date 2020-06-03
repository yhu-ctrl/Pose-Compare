import time

class FPS():
    __fps_time = 0

    @staticmethod
    def fps():
        fps = f"FPS:{(1.0 / (time.time() - FPS.__fps_time)):.2f}"
        FPS.__fps_time = time.time()

        return fps