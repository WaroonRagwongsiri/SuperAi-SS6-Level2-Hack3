from arduino.app_utils import *
import time
import threading
import os
import re
import sqlite3
import json
import urllib.error
import urllib.parse
import urllib.request
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
from score import MAX_SCORE, score_punch
from punch_model import TorchScriptPunchPredictor

# =========================
# UI STATE
# =========================
latest_msg = {
    "raw": "",
    "time": 0,
    "state": "resting",
    "countdown": 2.0,
    "phase_duration": 2.0,
    "punch_id": 0,
    "latest_score": 0.0,
    "max_score": MAX_SCORE,
    "metrics": {},
    "details": {},
    "cycle_time": 0.0
}

# =========================
# TIMED PUNCH DETECTION SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fight_scores.db")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
PUNCH_MODEL = TorchScriptPunchPredictor(MODEL_DIR)
ENV_PATH = os.path.join(BASE_DIR, ".env")
PUNCH_DETECTION_THRESHOLD = 0.65
PUNCH_MODEL.threshold = max(PUNCH_MODEL.threshold, PUNCH_DETECTION_THRESHOLD)
RESTING_SEC = 2.0
PUNCHING_SEC = 2.0
cycle_start_time = time.time()
timer_mode = "practice"

# Punch buffer
punch_buffer = []
last_state = "resting"
db_lock = threading.Lock()

api = FastAPI()


def load_env_file(path):
    if not os.path.isfile(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(ENV_PATH)

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


def extract_gemini_text(payload):
    parts = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text.strip())
    return "\n".join(parts).strip()


def fallback_punch_advice(result):
    details = result.get("details") or {}
    metrics = result.get("metrics") or {}
    weak_scores = sorted(
        (
            (key, float(value))
            for key, value in details.items()
            if isinstance(value, (int, float)) and key != "duration"
        ),
        key=lambda item: item[1]
    )
    weakest = weak_scores[0][0].replace("_", " ") if weak_scores else "straightness"
    straightness = metrics.get("straightness")
    snap = metrics.get("snap")

    lines = [
        "ยังไม่ได้ตั้งค่า Gemini API key ตอนนี้จึงใช้คำแนะนำพื้นฐานจากระบบ:",
        f"ควรฝึกเรื่อง {weakest} ก่อน",
        "หมัดตรงควรออกจากการ์ดเป็นเส้นตรง หมุนสะโพกและไหล่ด้านหลังไปพร้อมกัน แล้วรีบดึงหมัดกลับมาที่แก้ม",
    ]

    if isinstance(straightness, (int, float)) and straightness < 0.7:
        lines.append("แนวหมัดยังโค้งเล็กน้อย ให้เล็งผ่านจุดเดียวตรงหน้า และเก็บศอกให้อยู่หลังหมัด")

    if isinstance(snap, (int, float)) and snap < 0.65:
        lines.append("ถ้าอยากให้หมัดคมขึ้น ให้ผ่อนแรงก่อนออกหมัด เกร็งตอนกระแทก แล้วดึงกลับทันที")

    return "\n".join(lines)


def build_punch_prompt(result, mode):
    return f"""
คุณเป็นโค้ชมวยที่ช่วยผู้เริ่มต้นพัฒนาหมัด cross punch หรือหมัดตรง
ใช้คะแนนจากเซนเซอร์และ motion metrics ด้านล่างเพื่อให้คำแนะนำ
ตอบเป็นภาษาไทยเท่านั้น ใช้คำง่าย ๆ สั้น ชัดเจน เหมาะกับคนเพิ่งฝึก
อย่าวินิจฉัยอาการบาดเจ็บ ให้แนะนำเฉพาะเทคนิคและแบบฝึก

Mode: {mode}
Punch count: {result.get("punchCount", 0)}
Average score: {result.get("averageScore", 0)} out of {MAX_SCORE:g}
Score breakdown: {json.dumps(result.get("details") or {}, sort_keys=True)}
Motion metrics: {json.dumps(result.get("metrics") or {}, sort_keys=True)}

รูปแบบคำตอบ:
1. สรุปสั้น ๆ 1 ประโยคว่าหมัดนี้ควรปรับอะไรที่สุด
2. จุดที่ต้องแก้ 3 ข้อ แต่ละข้อให้มีคำสั่งที่ทำตามได้ทันที
3. แบบฝึก 30 วินาทีสำหรับรอบถัดไป
""".strip()


def request_gemini_advice(result, mode):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"advice": fallback_punch_advice(result), "source": "local"}

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
    prompt = build_punch_prompt(result, mode)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent?key="
        + urllib.parse.quote(api_key, safe="")
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 360,
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore") or str(exc)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {message[:300]}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}")

    advice = extract_gemini_text(payload)
    if not advice:
        raise HTTPException(status_code=502, detail="Gemini returned no coaching text")

    return {"advice": advice, "source": model}


def get_current_state():
    cycle_sec = RESTING_SEC + PUNCHING_SEC
    cycle_time = (time.time() - cycle_start_time) % cycle_sec

    if cycle_time < RESTING_SEC:
        return "resting", cycle_time, RESTING_SEC - cycle_time, RESTING_SEC

    punching_time = cycle_time - RESTING_SEC
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
            "gz_dps",
            "peak"
        ]
    )

    df["phase"] = "punch"

    try:
        model_samples = df[
            ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "peak"]
        ].to_dict("records")
        prediction = PUNCH_MODEL.predict_samples(model_samples)

        if not prediction.get("is_punch"):
            latest_msg["metrics"] = prediction
            latest_msg["details"] = {}
            print(
                "Punch ignored: model predicted rest "
                f"(prob={prediction.get('probability', 0.0):.3f})"
            )
            punch_buffer = []
            return

        result = score_punch(df, weights={"duration": 0.0})
        result.get("metrics", {}).pop("duration", None)
        result.get("scores", {}).pop("duration", None)
        result["metrics"]["punch_probability"] = prediction.get("probability", 0.0)

        score_val = result.get("overall", 0.0)
        detail_scores = result.get("scores", {})
        metrics = result.get("metrics", {})

        latest_msg["latest_score"] = score_val
        latest_msg["max_score"] = result.get("max_score", MAX_SCORE)
        latest_msg["metrics"] = metrics
        latest_msg["details"] = detail_scores
        latest_msg["punch_id"] += 1

        print(f">>>> RNN CONFIRMED PUNCH SCORED: {score_val} / {MAX_SCORE:g} <<<<")
        print("DETAILS:", detail_scores)
        print("METRICS:", metrics)

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

    if last_state == "resting" and current_state == "punching":
        print(">>>> TIMED PUNCH START <<<<")
        punch_buffer = []

    if last_state == "punching" and current_state == "resting":
        print(">>>> TIMED PUNCH END <<<<")
        score_current_punch()

    last_state = current_state

    if current_state == "punching":
        punch_buffer.append([
            now_ms,
            ax, ay, az,
            gx, gy, gz,
            peak
        ])

        print(f"STORE PUNCH DATA: countdown={countdown:.1f}s buffer={len(punch_buffer)}")


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

@api.post("/punch-advice")
async def punch_advice(request: Request):
    data = await request.json()
    result = data.get("result") or {}
    mode = str(data.get("mode") or "training")[:40]

    if not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="Invalid result payload")

    return request_gemini_advice(result, mode)

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
    return {"ok": True, "mode": timer_mode, "resting_sec": RESTING_SEC, "punching_sec": PUNCHING_SEC}

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
