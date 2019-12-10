import threading
import time


class WaitThread(threading.Thread):
    def run(self):
        self.stopped = False
        print("Thread started!")
        while not self.stopped:
            time.sleep(2)  # Pretend to work for a second
        print("Thread finished!")

    def stop(self):
        self.stopped = False
