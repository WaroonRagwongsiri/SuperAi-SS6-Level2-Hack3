from arduino.app_utils import *
import time
import threading
import os
import re
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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
    "countdown": 5.0,
    "phase_duration": 5.0,
    "punch_id": 0,
    "latest_score": 0.0,
    "max_score": MAX_SCORE,
    "details": {},
    "cycle_time": 0.0
}

# =========================
# TIMED PRACTICE SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fight_scores.db")
RESTING_SEC = 5.0
FIGHT_RESTING_SEC = 1.0
PUNCHING_SEC = 2.0
cycle_start_time = time.time()
timer_mode = "practice"

# Punch buffer
punch_buffer = []
last_state = "resting"
db_lock = threading.Lock()

api = FastAPI()

# Nano format:
# control=0 acc=0.02,-0.76,0.61 gyro=0.7,2.1,1.0 peak=0.19
LINE_RE = re.compile(
    r'control=(\d+)\s+'
    r'acc=([-\d.]+),([-\d.]+),([-\d.]+)\s+'
    r'gyro=([-\d.]+),([-\d.]+),([-\d.]+)\s+'
    r'peak=([-\d.]+)'
)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fight_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                total_punch INTEGER NOT NULL,
                average_score REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.commit()


def save_fight_score(player_name, total_punch, average_score):
    clean_name = str(player_name or "Optimizer01").strip()[:80] or "Optimizer01"
    punch_count = max(0, int(total_punch or 0))
    avg_score = max(0.0, float(average_score or 0))

    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO fight_scores (player_name, total_punch, average_score, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, punch_count, avg_score, int(time.time()))
            )
            conn.commit()


def get_leaderboard_rows(limit=10):
    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT player_name, total_punch, average_score, created_at
                FROM fight_scores
                ORDER BY average_score DESC, total_punch DESC, created_at ASC
                LIMIT ?
                """,
                (int(limit),)
            ).fetchall()

    return [
        {
            "name": row["player_name"],
            "total_punch": row["total_punch"],
            "average_score": round(row["average_score"], 1),
            "score": round(row["average_score"], 1),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_resting_sec():
    return FIGHT_RESTING_SEC if timer_mode == "fight" else RESTING_SEC


def get_current_state():
    resting_sec = get_resting_sec()
    cycle_sec = resting_sec + PUNCHING_SEC
    cycle_time = (time.time() - cycle_start_time) % cycle_sec

    if cycle_time < resting_sec:
        return "resting", cycle_time, resting_sec - cycle_time, resting_sec

    punching_time = cycle_time - resting_sec
    return "punching", cycle_time, PUNCHING_SEC - punching_time, PUNCHING_SEC


def update_ui_timer(current_state, cycle_time, countdown, phase_duration):
    latest_msg["state"] = current_state
    latest_msg["cycle_time"] = round(cycle_time, 2)
    latest_msg["countdown"] = round(max(0.0, countdown), 1)
    latest_msg["phase_duration"] = phase_duration


def score_current_punch():
    global punch_buffer, latest_msg

    if len(punch_buffer) < 5:
        print("Punch ignored: not enough data")
        punch_buffer = []
        return

    first_ms = punch_buffer[0][0]
    normalized_buffer = [
        [row[0] - first_ms, *row[1:]]
        for row in punch_buffer
    ]

    df = pd.DataFrame(
        normalized_buffer,
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
        latest_msg["punch_id"] += 1

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

    current_state, cycle_time, countdown, phase_duration = get_current_state()
    update_ui_timer(current_state, cycle_time, countdown, phase_duration)

    # =========================
    # STATE CHANGE DETECTION
    # =========================

    # RESTING -> PUNCHING
    if last_state == "resting" and current_state == "punching":
        print(">>>> TIMED PUNCH START <<<<")
        punch_buffer = []

    # PUNCHING -> RESTING
    if last_state == "punching" and current_state == "resting":
        print(">>>> TIMED PUNCH END <<<<")
        score_current_punch()

    last_state = current_state

    # =========================
    # STORE DATA ONLY DURING PUNCHING
    # =========================
    if current_state == "punching":
        punch_buffer.append([
            now_ms,
            ax, ay, az,
            gx, gy, gz
        ])

        print(f"STORE PUNCH DATA: countdown={countdown:.1f}s buffer={len(punch_buffer)}")

    else:
        # Resting state: do not save IMU data
        pass


Bridge.provide("serial_from_nano", serial_from_nano)

@api.get("/", response_class=FileResponse)
def home():
    return os.path.join(BASE_DIR, "index.html")

@api.get("/images/{filename}", response_class=FileResponse)
def image(filename: str):
    return os.path.join(BASE_DIR, "images", filename)

@api.get("/latest")
def latest():
    current_state, cycle_time, countdown, phase_duration = get_current_state()
    update_ui_timer(current_state, cycle_time, countdown, phase_duration)
    latest_msg["time"] = time.time()
    return latest_msg

@api.post("/fight-results")
async def fight_results(request: Request):
    data = await request.json()
    save_fight_score(
        data.get("name"),
        data.get("total_punch"),
        data.get("average_score")
    )
    return {"ok": True}

@api.get("/leaderboard-data")
def leaderboard_data():
    return {"rows": get_leaderboard_rows()}

@api.post("/timer-mode/{mode}")
def set_timer_mode(mode: str):
    global timer_mode, cycle_start_time, last_state, punch_buffer

    if mode not in ("practice", "fight"):
        raise HTTPException(status_code=400, detail="Invalid timer mode")

    timer_mode = mode
    cycle_start_time = time.time()
    last_state = "resting"
    punch_buffer = []
    current_state, cycle_time, countdown, phase_duration = get_current_state()
    update_ui_timer(current_state, cycle_time, countdown, phase_duration)
    return {"ok": True, "mode": timer_mode, "resting_sec": get_resting_sec()}

@api.get("/{page_path:path}", response_class=FileResponse)
def html_page(page_path: str):
    if not page_path.endswith(".html"):
        raise HTTPException(status_code=404, detail="Not Found")

    requested_path = os.path.abspath(os.path.join(BASE_DIR, page_path))
    base_path = os.path.abspath(BASE_DIR)

    if not requested_path.startswith(base_path + os.sep):
        raise HTTPException(status_code=404, detail="Not Found")

    if not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Not Found")

    return requested_path

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
init_db()
api_thread.start()

App.run(user_loop=loop)
