from arduino.app_utils import *
import time
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import os

latest_msg = {"raw": "", "time": 0}

api = FastAPI()

def serial_from_nano(data):
    global latest_msg
    latest_msg = {"raw": str(data), "time": time.time()}
    print("FROM MCU:", data)

Bridge.provide("serial_from_nano", serial_from_nano)

@api.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>UNO Q Webcam</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0a0a0c; color: #e8e8f0;
      font-family: 'Segoe UI', sans-serif;
      min-height: 100vh; display: flex; flex-direction: column;
      align-items: center; padding: 24px 16px 40px; gap: 16px;
    }
    h1 { font-size: 32px; font-weight: 900; color: #ffd400; letter-spacing: 0.04em; }
    .cam-wrap {
      position: relative; width: 100%; max-width: 960px;
      aspect-ratio: 16/9; background: #000;
      border-radius: 14px; overflow: hidden; border: 1px solid #222;
    }
    #webcam { width: 100%; height: 100%; object-fit: cover; display: block; }
    .placeholder {
      position: absolute; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 14px; background: #111;
    }
    .placeholder p { font-size: 13px; color: #555; font-family: monospace; }
    #openBtn {
      padding: 10px 22px; background: #ffd400; color: #111;
      border: none; border-radius: 8px; font-weight: 700;
      font-size: 15px; cursor: pointer;
    }
    #openBtn:hover { opacity: .85; }
    .scorebox {
      position: absolute; top: 14px; right: 14px;
      background: rgba(0,0,0,.78); border: 2px solid #ffd400;
      border-radius: 12px; padding: 10px 18px 14px;
      text-align: center; z-index: 20; min-width: 100px;
    }
    .scorebox .lbl { font-size: 10px; letter-spacing: 0.18em; color: #666; text-transform: uppercase; }
    .scorebox .num {
      font-size: 56px; font-weight: 900; color: #ffd400; line-height: 1;
      text-shadow: 0 0 20px rgba(255,212,0,.45);
      transition: transform .12s cubic-bezier(.34,1.56,.64,1);
    }
    .scorebox .num.bump { transform: scale(1.3); }
    .signal {
      width: 100%; max-width: 960px; background: #111;
      border: 1px solid #222; border-radius: 10px;
      padding: 12px 16px; display: flex; align-items: center; gap: 12px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #333; flex-shrink: 0; transition: background .3s; }
    .dot.live { background: #3dff8f; box-shadow: 0 0 7px #3dff8f99; }
    .raw { font-family: monospace; font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ts { font-family: monospace; font-size: 11px; color: #555; flex-shrink: 0; }
  </style>
</head>
<body>
  <h1>UNO Q — Webcam</h1>
  <div class="cam-wrap">
    <video id="webcam" autoplay playsinline muted></video>
    <div class="placeholder" id="ph">
      <p>No camera feed</p>
      <button id="openBtn" onclick="openCam()">Open Webcam</button>
    </div>
    <div class="scorebox">
      <div class="lbl">Score</div>
      <div class="num" id="score">0</div>
    </div>
  </div>
  <div class="signal">
    <div class="dot" id="dot"></div>
    <div class="raw" id="raw">Waiting for Nano signal…</div>
    <div class="ts" id="ts">—</div>
  </div>
  <script>
    async function openCam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const v = document.getElementById('webcam');
        v.srcObject = stream;
        await v.play();
        document.getElementById('ph').style.display = 'none';
      } catch (e) {
        alert('Camera error: ' + e.name + ' — ' + e.message);
      }
    }
    let score = 0, lastRaw = '';
    function bump() {
      score++;
      const el = document.getElementById('score');
      el.textContent = score;
      el.classList.remove('bump');
      void el.offsetWidth;
      el.classList.add('bump');
      setTimeout(() => el.classList.remove('bump'), 180);
    }
    async function poll() {
      try {
        const r = await fetch('/latest', { cache: 'no-store' });
        const d = await r.json();
        const raw = d.raw || '';
        document.getElementById('dot').classList.toggle('live', !!raw);
        document.getElementById('raw').textContent = raw || 'Waiting for Nano signal…';
        if (d.time) document.getElementById('ts').textContent = new Date(d.time * 1000).toLocaleTimeString();
        if (raw.includes('control=1') && raw !== lastRaw) { lastRaw = raw; bump(); }
      } catch (_) {}
    }
    setInterval(poll, 300);
    poll();
  </script>
</body>
</html>"""

@api.get("/latest")
def latest():
    return latest_msg

# Always relative to this file's location — never depends on cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(BASE_DIR, "cert.pem")
KEY  = os.path.join(BASE_DIR, "key.pem")

def run_api():
    if not os.path.exists(CERT) or not os.path.exists(KEY):
        print(f"ERROR: cert.pem or key.pem not found in {BASE_DIR}")
        print("Run: openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'")
        return
    config = uvicorn.Config(
        app=api,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=CERT,
        ssl_keyfile=KEY,
        log_level="info",
        access_log=False,
    )
    uvicorn.Server(config).run()

def loop():
    time.sleep(0.01)

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

App.run(user_loop=loop)