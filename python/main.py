from arduino.app_utils import *
import time
import threading
import os
import re
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from score import MAX_SCORE, score_punch

# =========================
# UI STATE
# =========================
latest_msg = {
    "raw": "",
    "time": 0,
    "state": "resting",
    "latest_score": 0.0,
    "max_score": MAX_SCORE,
    "details": {},
    "cycle_time": 0.0
}

# =========================
# FIXED TIME FRAME SETTINGS
# =========================
RESTING_SEC = 4.0
PUNCHING_SEC = 2.0
CYCLE_SEC = RESTING_SEC + PUNCHING_SEC

# This is when the repeating cycle starts
cycle_start_time = time.time()

# Punch buffer
punch_buffer = []
last_state = "punching"

api = FastAPI()

# Nano format:
# control=0 acc=0.02,-0.76,0.61 gyro=0.7,2.1,1.0 peak=0.19
LINE_RE = re.compile(
    r'control=(\d+)\s+'
    r'acc=([-\d.]+),([-\d.]+),([-\d.]+)\s+'
    r'gyro=([-\d.]+),([-\d.]+),([-\d.]+)\s+'
    r'peak=([-\d.]+)'
)

def get_current_state():
    """
    UNO Q decides state by fixed time window:
    0-2 sec  = punching
    2-6 sec  = resting
    repeat
    """
    now = time.time()
    cycle_time = (now - cycle_start_time) % CYCLE_SEC

    if cycle_time < PUNCHING_SEC:
        return "punching", cycle_time
    else:
        return "resting", cycle_time


def score_current_punch():
    global punch_buffer, latest_msg

    if len(punch_buffer) < 5:
        print("Punch ignored: not enough data")
        punch_buffer = []
        return

    df = pd.DataFrame(
        punch_buffer,
        columns=[
            "phase_elapsed_ms",
            "ax_g",
            "ay_g",
            "az_g",
            "gx_dps",
            "gy_dps",
            "gz_dps"
        ]
    )

    df["phase"] = "punch"

    try:
        result = score_punch(df)

        score_val = result.get("overall", 0.0)
        detail_scores = result.get("scores", {})

        latest_msg["latest_score"] = score_val
        latest_msg["max_score"] = result.get("max_score", MAX_SCORE)
        latest_msg["details"] = detail_scores

        print(f">>>> PUNCH SCORED: {score_val} / {MAX_SCORE:g} <<<<")
        print("DETAILS:", detail_scores)

    except Exception as e:
        print(f"Error scoring punch: {e}")

    punch_buffer = []


def serial_from_nano(data):
    global latest_msg, punch_buffer, last_state

    raw_str = data.decode("utf-8", errors="ignore").strip() if isinstance(data, bytes) else str(data).strip()

    latest_msg["raw"] = raw_str
    latest_msg["time"] = time.time()

    print("FROM MCU:", raw_str)

    match = LINE_RE.search(raw_str)
    if not match:
        print("Regex not match")
        return

    now_ms = time.time() * 1000

    # control and peak are parsed but NOT used for state
    ctrl = int(match.group(1))

    ax = float(match.group(2))
    ay = float(match.group(3))
    az = float(match.group(4))

    gx = float(match.group(5))
    gy = float(match.group(6))
    gz = float(match.group(7))

    peak = float(match.group(8))

    current_state, cycle_time = get_current_state()

    latest_msg["state"] = current_state
    latest_msg["cycle_time"] = round(cycle_time, 2)

    # =========================
    # STATE CHANGE DETECTION
    # =========================

    # RESTING -> PUNCHING
    if last_state == "resting" and current_state == "punching":
        print(">>>> TIME WINDOW PUNCH START <<<<")
        punch_buffer = []

    # PUNCHING -> RESTING
    if last_state == "punching" and current_state == "resting":
        print(">>>> TIME WINDOW PUNCH END <<<<")
        score_current_punch()

    last_state = current_state

    # =========================
    # STORE DATA ONLY DURING PUNCHING
    # =========================
    if current_state == "punching":
        punch_elapsed_ms = cycle_time * 1000

        punch_buffer.append([
            punch_elapsed_ms,
            ax, ay, az,
            gx, gy, gz
        ])

        print(f"STORE PUNCH DATA: t={punch_elapsed_ms:.1f} ms buffer={len(punch_buffer)}")

    else:
        # Resting state: do not save IMU data
        pass


Bridge.provide("serial_from_nano", serial_from_nano)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@api.get("/", response_class=FileResponse)
def home():
    return os.path.join(BASE_DIR, "index.html")

@api.get("/images/{filename}", response_class=FileResponse)
def image(filename: str):
    return os.path.join(BASE_DIR, "images", filename)

@api.get("/latest")
def latest():
    # Update state even if no new serial data arrives
    current_state, cycle_time = get_current_state()
    latest_msg["state"] = current_state
    latest_msg["cycle_time"] = round(cycle_time, 2)
    latest_msg["time"] = time.time()
    return latest_msg

CERT = os.path.join(BASE_DIR, "cert.pem")
KEY = os.path.join(BASE_DIR, "key.pem")

def run_api():
    if not os.path.exists(CERT) or not os.path.exists(KEY):
        print(f"ERROR: cert.pem or key.pem not found in {BASE_DIR}")
        return

    config = uvicorn.Config(
        app=api,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=CERT,
        ssl_keyfile=KEY,
        log_level="info",
        access_log=False
    )

    uvicorn.Server(config).run()


def loop():
    time.sleep(0.01)


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

App.run(user_loop=loop)
