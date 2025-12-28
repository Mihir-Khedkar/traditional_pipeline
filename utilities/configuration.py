from datetime import datetime
class Logger():
    def clog(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
        print(f"[{timestamp}] {message}")

    def minilogger(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S:%f")
        print(f"           [{timestamp}] {message}")