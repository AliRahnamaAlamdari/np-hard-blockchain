# utils/logs.py
class BlockEventLog:
    def __init__(self, height: int):
        self.height = height
        self.events = {}

    def add(self, key, value):
        self.events[key] = value

    def print(self):
        print(f"\n===== BLOCK HEIGHT {self.height} =====")
        for k, v in self.events.items():
            print(f"{k}: {v}")
        print("------------------------------------------")
