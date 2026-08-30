import subprocess
import time
import sys
import os
import signal

def run_worker_with_watchdog(timeout_sec: int):
    # This is a simplified watchdog. A full implementation would use resource module
    # or cgroups to enforce memory limits (memory_mb).
    print(f"Starting worker daemon with {timeout_sec}s timeout watchdog...")
    
    # Start the daemon
    worker_path = os.path.join(os.path.dirname(__file__), "daemon.py")
    process = subprocess.Popen([sys.executable, worker_path])
    
    start_time = time.time()
    try:
        while True:
            if process.poll() is not None:
                print("Worker exited on its own.")
                break
                
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                print(f"Timeout of {timeout_sec}s exceeded. Terminating worker.")
                process.send_signal(signal.SIGKILL)
                break
                
            time.sleep(1)
    except KeyboardInterrupt:
        process.terminate()
        
if __name__ == "__main__":
    run_worker_with_watchdog(timeout_sec=300) # Default 5 min for the daemon itself
