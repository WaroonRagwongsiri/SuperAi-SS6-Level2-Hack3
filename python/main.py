from arduino.app_utils import *
import time
import threading
import os
import re
import collections
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from score import score_punch # Imports from your score.py

# Track the state for the web UI
latest_msg = {"raw": "", "time": 0, "latest_score": 0.0, "details": {}}

# Create a rolling buffer that will store our last 5 seconds of data
rolling_buffer = collections.deque()
is_punching = False
punch_start_time = 0

api = FastAPI()

def serial_from_nano(data):
    global latest_msg, is_punching, punch_start_time, rolling_buffer
    
    # Safely convert incoming data to string
    raw_str = data.decode('utf-8').strip() if isinstance(data, bytes) else str(data).strip()
    
    # Keep updating the raw feed for the UI
    latest_msg["raw"] = raw_str
    latest_msg["time"] = time.time()
    print("FROM MCU:", raw_str)

    # Updated Regex to extract: control, ax, ay, az, gx, gy, gz, peak
    match = re.search(r'control=(\d)\s+acc=([-\d.]+),([-\d.]+),([-\d.]+)\s+gyro=([-\d.]+),([-\d.]+),([-\d.]+)\s+peak=([-\d.]+)', raw_str)
    
    if match:
        now_ms = time.time() * 1000
        ctrl = int(match.group(1))
        ax, ay, az = map(float, match.group(2, 3, 4))
        gx, gy, gz = map(float, match.group(5, 6, 7))
        peak = float(match.group(8)) 
        
        # 1. Add current reading to our buffer
        rolling_buffer.append([now_ms, ax, ay, az, gx, gy, gz])
        
        # 2. Trim the buffer so it only keeps the last 5000 milliseconds (5 seconds)
        while len(rolling_buffer) > 0 and (now_ms - rolling_buffer[0][0]) > 5000:
            rolling_buffer.popleft()
        
        # 3. Punch Detection Logic
        if ctrl == 1:
            if not is_punching:
                # A new punch just started
                is_punching = True
                punch_start_time = now_ms
                
        elif ctrl == 0 and is_punching:
            # Punch finished
            is_punching = False
            
            # Extract the data from our 5-second buffer.
            # We grab data starting 150ms BEFORE the punch started so `score_punch`
            # has stationary data to calibrate the gyro and gravity drift.
            punch_data = [row for row in rolling_buffer if row[0] >= (punch_start_time - 150)]
            
            # Make sure we actually collected enough data points
            if len(punch_data) >= 5:
                df = pd.DataFrame(punch_data, columns=[
                    'phase_elapsed_ms', 'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'
                ])
                df['phase'] = 'punch' # Required by your scoring function
                
                try:
                    # Score it!
                    result = score_punch(df)
                    score_val = result.get('overall', 0.0)
                    
                    # Send both the overall score AND the breakdown to the UI
                    latest_msg["latest_score"] = score_val
                    latest_msg["details"] = result.get('scores', {})
                    
                    print(f">>>> PUNCH SCORED: {score_val} / 100 <<<<")
                except Exception as e:
                    print(f"Error scoring punch: {e}")

Bridge.provide("serial_from_nano", serial_from_nano)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@api.get("/", response_class=FileResponse)
def home():
    # Serve the external HTML file
    return os.path.join(BASE_DIR, "index.html")

@api.get("/latest")
def latest():
    return latest_msg

CERT = os.path.join(BASE_DIR, "cert.pem")
KEY  = os.path.join(BASE_DIR, "key.pem")

def run_api():
    if not os.path.exists(CERT) or not os.path.exists(KEY):
        print(f"ERROR: cert.pem or key.pem not found in {BASE_DIR}")
        return
    config = uvicorn.Config(app=api, host="0.0.0.0", port=8000, ssl_certfile=CERT, ssl_keyfile=KEY, log_level="info", access_log=False)
    uvicorn.Server(config).run()

def loop():
    time.sleep(0.01)

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

App.run(user_loop=loop)