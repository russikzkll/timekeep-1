import http.server
import socketserver
import json
import os
import logging
import traceback
import ctypes
import ctypes.util
import platform
from datetime import datetime, time, timezone, timedelta
from urllib.parse import urlparse, parse_qs
import base64
import pickle
import numpy as np
import io

PORT = int(os.getenv("PORT", "8000"))
DATA_FILE = 'attendance.json'
FACES_FILE = 'faces_data.pkl'
WORK_START = time(14, 0)
WORK_END = time(17, 30)
ADMIN_PASSWORD = 'admin123'
FACE_DISTANCE_THRESHOLD = 0.40


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [timekeep-web] %(message)s'
)
logger = logging.getLogger('timekeep.web')


ATYRAU_TZ = timezone(timedelta(hours=5), "Asia/Atyrau")


def now_atyrau() -> datetime:
    return datetime.now(ATYRAU_TZ)


os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

DEEPFACE_IMPORT_ERROR = ''
try:
    from deepface import DeepFace
    FACE_RECOGNITION_AVAILABLE = True
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False
    DEEPFACE_IMPORT_ERROR = str(e)
    logger.exception('DeepFace import failed during startup')


def runtime_diagnostics() -> dict:
    libraries = {}
    for lib in ('GL', 'glib-2.0', 'X11', 'Xext', 'Xrender', 'SM'):
        found = ctypes.util.find_library(lib)
        libraries[lib] = found

    gl_load_error = ''
    gl_load_ok = False
    try:
        ctypes.CDLL('libGL.so.1')
        gl_load_ok = True
    except OSError as e:
        gl_load_error = str(e)

    return {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'face_recognition_available': FACE_RECOGNITION_AVAILABLE,
        'deepface_import_error': DEEPFACE_IMPORT_ERROR,
        'library_lookup': libraries,
        'libGL_load_ok': gl_load_ok,
        'libGL_load_error': gl_load_error,
    }

MAIN_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Time Keeper</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #F4EFE6;
  --card: #FFFFFF;
  --accent: #C9A96E;
  --accent-dark: #A8864F;
  --text: #3D3530;
  --muted: #7A726D;
  --border: #E8E2D9;
  --success: #5A8F7B;
  --error: #C0544A;
  --late: #D4856A;
  --vacation: #7A9E88;
  --absent: #9E9790;
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: var(--bg); min-height: 100vh; display: flex; justify-content: center;
       align-items: flex-start; padding: 2rem 1rem; }
.card { background: var(--card); border-radius: 24px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.06);
        padding: 2.5rem; width: 100%; max-width: 560px; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.logo { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.5px; }
.timer { font-size: 1.1rem; color: var(--accent); font-weight: 500; font-variant-numeric: tabular-nums; }
.cam-wrap { position: relative; background: #1a1a1a; border-radius: 16px;
            overflow: hidden; aspect-ratio: 4/3; margin-bottom: 1.25rem; }
#video { width: 100%; height: 100%; object-fit: cover; display: block; }
#canvas { display: none; }
.cam-overlay { position: absolute; inset: 0; display: flex; flex-direction: column;
               align-items: center; justify-content: center; color: #fff; gap: 0.5rem; }
.cam-overlay p { opacity: 0.5; font-size: 0.9rem; }
.btn { border: none; border-radius: 12px; cursor: pointer; font-weight: 500;
       font-size: 0.95rem; padding: 0.85rem 1.4rem; transition: all 0.2s; }
.btn-primary { background: var(--accent); color: #fff; width: 100%; }
.btn-primary:hover { background: var(--accent-dark); transform: translateY(-1px); }
.btn-primary:disabled { background: #ccc; cursor: not-allowed; transform: none; }
.btn-secondary { background: #F0EDE7; color: var(--muted); }
.btn-secondary:hover { background: var(--border); }
.btn-sm { padding: 0.6rem 1rem; font-size: 0.85rem; }
.btn-danger { background: #fde8e6; color: var(--error); }
.btn-danger:hover { background: #f5c6c2; }
#stepScan, #stepStatus, #stepDone { display: none; }
#stepScan.active, #stepStatus.active, #stepDone.active { display: block; }
.status-picker { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; }
.status-card { background: #FAFAF8; border: 2px solid var(--border); border-radius: 14px;
               padding: 1.2rem 0.75rem; text-align: center; cursor: pointer; transition: all 0.2s; }
.status-card:hover { border-color: var(--accent); background: rgba(201,169,110,0.07); }
.status-card .label { font-size: 0.85rem; font-weight: 500; color: var(--text); }
.greeting { font-size: 1.15rem; font-weight: 600; color: var(--text);
            margin-bottom: 1.5rem; text-align: center; }
.done-box { text-align: center; padding: 1rem 0; }
.done-box .done-name { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.3rem; }
.done-box .done-status { font-size: 1rem; color: var(--muted); margin-bottom: 1.5rem; }
.done-box.late .done-status { color: var(--late); }
.done-box.vacation .done-status { color: var(--vacation); }
.done-box.absent .done-status { color: var(--absent); }
.msg { padding: 0.85rem 1.1rem; border-radius: 10px; font-size: 0.9rem;
       text-align: center; margin-bottom: 1rem; }
.msg-error { background: #fde8e6; color: var(--error); }
.msg-success { background: #e6f4ef; color: var(--success); }
.msg-info { background: #F0EDE7; color: var(--muted); }
.section-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.2px;
                 color: var(--muted); font-weight: 500; margin: 2rem 0 1rem; }
.entry { display: flex; justify-content: space-between; align-items: center;
         padding: 1rem 1.2rem; border-radius: 14px; background: #FAFAF8;
         border: 1px solid var(--border); margin-bottom: 0.75rem; }
.entry .ename { font-weight: 500; display: flex; align-items: center; gap: 0.4rem; }
.entry .ename .badge { font-size: 0.7rem; background: #e6f4ef; color: var(--success);
                       padding: 0.15rem 0.45rem; border-radius: 20px; }
.entry .einfo { text-align: right; }
.entry .etime { font-size: 0.8rem; color: var(--muted); display: block; }
.entry .estats { font-size: 0.82rem; font-weight: 500; }
.ok { color: var(--success); }
.late2 { color: var(--late); }
.vac { color: var(--vacation); }
.abs { color: var(--absent); }
.empty-list { text-align: center; color: var(--muted); padding: 2rem; font-size: 0.9rem; }
.admin-link { display: block; text-align: center; margin-top: 2rem; color: var(--muted);
              font-size: 0.8rem; text-decoration: none; opacity: 0.5; }
.admin-link:hover { opacity: 1; }
@media(max-width:480px) { .status-picker { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="card">
  <header>
    <div class="logo">Time Keeper</div>
    <div id="countdown" class="timer">--:--</div>
  </header>

  <div id="stepScan" class="active">
    <div class="cam-wrap">
      <video id="video" autoplay playsinline></video>
      <canvas id="canvas"></canvas>
      <div id="camPlaceholder" class="cam-overlay">
        <p>Нажмите кнопку, чтобы включить камеру</p>
      </div>
    </div>
    <div id="scanMsg"></div>
    <button id="mainBtn" class="btn btn-primary" onclick="handleMainBtn()">Включить камеру</button>
  </div>

  <div id="stepStatus">
    <div id="greetingText" class="greeting"></div>
    <div class="status-picker">
      <div class="status-card" onclick="submitStatus('present')">
        <span class="label">Я пришёл</span>
      </div>
      <div class="status-card" onclick="submitStatus('absent')">
        <span class="label">Не приду</span>
      </div>
      <div class="status-card" onclick="submitStatus('vacation')">
        <span class="label">В отпуске</span>
      </div>
    </div>
    <br>
    <button class="btn btn-secondary btn-sm" style="width:100%" onclick="resetToScan()">Назад</button>
  </div>

  <div id="stepDone">
    <div id="doneBox" class="done-box">
      <div class="done-name" id="doneName"></div>
      <div class="done-status" id="doneStatusText"></div>
      <button class="btn btn-primary" onclick="resetToScan()">Готово</button>
    </div>
  </div>

  <div class="section-title">Сегодня</div>
  <div id="attendanceList"></div>

  <a href="/admin" class="admin-link">Панель администратора</a>
</div>

<script>
let videoStream = null;
let cameraActive = false;
let recognizedName = null;
let btnState = 'start';

function setStep(step) {
  ['stepScan','stepStatus','stepDone'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  document.getElementById('step' + step).classList.add('active');
}

function showMsg(elementId, text, type) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.className = 'msg msg-' + (type || 'info');
  el.innerHTML = text;
  el.style.display = text ? 'block' : 'none';
}

async function handleMainBtn() {
  if (btnState === 'start') {
    await startCamera();
  } else if (btnState === 'capture') {
    await captureFace();
  } else if (btnState === 'retry') {
    showMsg('scanMsg', '', 'info');
    setBtnState('capture');
  }
}

function setBtnState(state) {
  const btn = document.getElementById('mainBtn');
  btnState = state;
  if (state === 'start') {
    btn.textContent = 'Включить камеру';
    btn.disabled = false;
  } else if (state === 'capture') {
    btn.textContent = 'Сфотографировать и войти';
    btn.disabled = false;
  } else if (state === 'loading') {
    btn.textContent = 'Распознаю лицо...';
    btn.disabled = true;
  } else if (state === 'retry') {
    btn.textContent = 'Попробовать снова';
    btn.disabled = false;
  }
}

async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    const video = document.getElementById('video');
    video.srcObject = videoStream;
    document.getElementById('camPlaceholder').style.display = 'none';
    cameraActive = true;
    setBtnState('capture');
    showMsg('scanMsg', 'Встаньте перед камерой и нажмите кнопку', 'info');
  } catch (err) {
    showMsg('scanMsg', 'Не удалось открыть камеру: ' + err.message, 'error');
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
  }
  const video = document.getElementById('video');
  video.srcObject = null;
  document.getElementById('camPlaceholder').style.display = 'flex';
  cameraActive = false;
}

async function captureFace() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const imageData = canvas.toDataURL('image/jpeg', 0.9);

  setBtnState('loading');
  showMsg('scanMsg', 'Распознаю...', 'info');

  try {
    const resp = await fetch('/api/identify-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    const data = await resp.json();

    if (data.success) {
      recognizedName = data.name;
      stopCamera();
      showStatusStep(data.name, data.already_checked_in);
    } else {
      showMsg('scanMsg', data.message, 'error');
      setBtnState('retry');
    }
  } catch (err) {
    showMsg('scanMsg', 'Ошибка сети: ' + err.message, 'error');
    setBtnState('retry');
  }
}

function showStatusStep(name, alreadyCheckedIn) {
  if (alreadyCheckedIn) {
    showDoneStep(name, 'Уже отмечен сегодня', 'ok');
    return;
  }
  document.getElementById('greetingText').textContent = 'Привет, ' + name + '!';
  setStep('Status');
}

async function submitStatus(statusType) {
  if (!recognizedName) return;
  try {
    const resp = await fetch('/api/face-checkin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: recognizedName, status: statusType })
    });
    const data = await resp.json();
    if (data.error) {
      alert(data.error);
    } else {
      let cssClass = statusType === 'vacation' ? 'vacation'
                   : statusType === 'absent' ? 'absent'
                   : data.is_late ? 'late' : '';
      showDoneStep(recognizedName, data.status, cssClass);
      fetchStatus();
    }
  } catch (err) {
    alert('Ошибка: ' + err.message);
  }
}

function showDoneStep(name, statusText, cssClass) {
  document.getElementById('doneName').textContent = name;
  document.getElementById('doneStatusText').textContent = statusText;
  const box = document.getElementById('doneBox');
  box.className = 'done-box ' + (cssClass || '');
  setStep('Done');
  fetchStatus();
}

function resetToScan() {
  recognizedName = null;
  stopCamera();
  showMsg('scanMsg', '', 'info');
  setBtnState('start');
  setStep('Scan');
}

async function fetchStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    const el = document.getElementById('attendanceList');
    if (!data.length) {
      el.innerHTML = '<div class="empty-list">Пока никого нет...</div>';
      return;
    }
    el.innerHTML = data.map(e => {
      let cls = e.is_late ? 'late2' : e.type === 'vacation' ? 'vac'
              : e.type === 'absent' ? 'abs' : 'ok';
      return `<div class="entry">
        <div class="ename">${e.name} <span class="badge">верифицирован</span></div>
        <div class="einfo">
          <span class="etime">${e.time}</span>
          <span class="estats ${cls}">${e.status}</span>
        </div>
      </div>`;
    }).join('');
  } catch (err) { console.error(err); }
}

async function updateTimer() {
  try {
    const resp = await fetch('/api/time_left');
    const data = await resp.json();
    const el = document.getElementById('countdown');
    if (data.finished) {
      el.textContent = 'День окончен';
      el.style.color = '#aaa';
    } else {
      el.textContent = 'До конца: ' + data.time_left;
      el.style.color = 'var(--accent)';
    }
  } catch(e) {}
}

fetchStatus();
updateTimer();
setInterval(updateTimer, 1000);
setInterval(fetchStatus, 20000);
</script>
</body>
</html>
"""

ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Time Keeper — Администратор</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root { --bg:#F4EFE6; --card:#fff; --accent:#C9A96E; --accent-dark:#A8864F;
        --text:#3D3530; --muted:#7A726D; --border:#E8E2D9;
        --success:#5A8F7B; --error:#C0544A; }
* { box-sizing:border-box; margin:0; padding:0; font-family:'Inter',sans-serif; }
body { background:var(--bg); min-height:100vh; display:flex; justify-content:center;
       align-items:flex-start; padding:2rem 1rem; }
.card { background:var(--card); border-radius:24px; box-shadow:0 8px 40px rgba(0,0,0,.06);
        padding:2.5rem; width:100%; max-width:620px; }
h1 { font-size:1.3rem; font-weight:600; margin-bottom:0.25rem; }
.subtitle { color:var(--muted); font-size:0.9rem; margin-bottom:2rem; }
.section-title { font-size:.75rem; text-transform:uppercase; letter-spacing:1.2px;
                 color:var(--muted); font-weight:500; margin:2rem 0 1rem; }
input[type=text], input[type=password] {
  width:100%; padding:.9rem 1.2rem; border:1px solid var(--border); border-radius:12px;
  font-size:1rem; outline:none; transition:all .2s; background:#FAFAF8; margin-bottom:.75rem; }
input:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(201,169,110,.12); }
.btn { border:none; border-radius:12px; cursor:pointer; font-weight:500; font-size:.95rem;
       padding:.85rem 1.4rem; transition:all .2s; }
.btn-primary { background:var(--accent); color:#fff; width:100%; }
.btn-primary:hover { background:var(--accent-dark); transform:translateY(-1px); }
.btn-primary:disabled { background:#ccc; cursor:not-allowed; transform:none; }
.btn-danger { background:#fde8e6; color:var(--error); padding:.5rem 1rem; font-size:.8rem; }
.btn-danger:hover { background:#f5c6c2; }
.cam-wrap { background:#1a1a1a; border-radius:16px; overflow:hidden; aspect-ratio:4/3;
            margin-bottom:1rem; }
#adminVideo { width:100%; height:100%; object-fit:cover; display:block; }
.msg { padding:.85rem 1.1rem; border-radius:10px; font-size:.9rem; text-align:center;
       margin-bottom:1rem; display:none; }
.msg-error { background:#fde8e6; color:var(--error); }
.msg-success { background:#e6f4ef; color:var(--success); }
.msg-info { background:#F0EDE7; color:var(--muted); }
.emp-row { display:flex; align-items:center; justify-content:space-between;
           padding:1rem 1.2rem; border-radius:14px; background:#FAFAF8;
           border:1px solid var(--border); margin-bottom:.6rem; }
.emp-name { font-weight:500; }
#adminSection { display:none; }
.back-link { display:block; text-align:center; margin-top:1.5rem; color:var(--muted);
             font-size:.85rem; text-decoration:none; }
.back-link:hover { color:var(--text); }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }
@media(max-width:480px) { .two-col { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="card">
  <h1>Панель администратора</h1>
  <p class="subtitle">Управление сотрудниками и регистрация лиц</p>

  <div id="loginSection">
    <div class="section-title">Вход</div>
    <input type="password" id="pwInput" placeholder="Пароль администратора..."
           onkeydown="if(event.key==='Enter')adminLogin()">
    <div id="loginMsg" class="msg"></div>
    <button class="btn btn-primary" onclick="adminLogin()">Войти</button>
  </div>

  <div id="adminSection">
    <div class="section-title">Зарегистрировать сотрудника</div>
    <input type="text" id="regName" placeholder="Имя сотрудника...">
    <div class="cam-wrap">
      <video id="adminVideo" autoplay playsinline></video>
    </div>
    <div id="regMsg" class="msg"></div>
    <div class="two-col">
      <button id="adminCamBtn" class="btn btn-primary" onclick="toggleAdminCam()">Включить камеру</button>
      <button id="adminCaptureBtn" class="btn btn-primary" disabled onclick="captureAndRegister()">
        Сфотографировать и зарегистрировать
      </button>
    </div>

    <div class="section-title">Зарегистрированные сотрудники</div>
    <div id="employeeList"></div>
  </div>

  <a href="/" class="back-link">Вернуться к главной</a>
</div>

<script>
let adminPwd = '';
let adminStream = null;
let adminCamOn = false;

function showMsg(id, text, type) {
  const el = document.getElementById(id);
  el.className = 'msg msg-' + (type || 'info');
  el.textContent = text;
  el.style.display = text ? 'block' : 'none';
}

async function adminLogin() {
  const pw = document.getElementById('pwInput').value;
  const resp = await fetch('/api/admin/auth', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({password: pw})
  });
  const data = await resp.json();
  if (data.success) {
    adminPwd = pw;
    document.getElementById('loginSection').style.display = 'none';
    document.getElementById('adminSection').style.display = 'block';
    loadEmployees();
  } else {
    showMsg('loginMsg', 'Неверный пароль', 'error');
  }
}

async function toggleAdminCam() {
  if (!adminCamOn) {
    try {
      adminStream = await navigator.mediaDevices.getUserMedia({ video: true });
      document.getElementById('adminVideo').srcObject = adminStream;
      adminCamOn = true;
      document.getElementById('adminCamBtn').textContent = 'Выключить камеру';
      document.getElementById('adminCaptureBtn').disabled = false;
    } catch(e) { showMsg('regMsg', e.message, 'error'); }
  } else {
    adminStream.getTracks().forEach(t => t.stop());
    document.getElementById('adminVideo').srcObject = null;
    adminCamOn = false;
    document.getElementById('adminCamBtn').textContent = 'Включить камеру';
    document.getElementById('adminCaptureBtn').disabled = true;
  }
}

async function captureAndRegister() {
  const name = document.getElementById('regName').value.trim();
  if (!name) { showMsg('regMsg', 'Введите имя сотрудника', 'error'); return; }
  if (!adminCamOn) { showMsg('regMsg', 'Включите камеру', 'error'); return; }

  const video = document.getElementById('adminVideo');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const imageData = canvas.toDataURL('image/jpeg', 0.9);

  showMsg('regMsg', 'Регистрирую...', 'info');
  const resp = await fetch('/api/admin/register-face', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ name, image: imageData, password: adminPwd })
  });
  const data = await resp.json();
  if (data.success) {
    showMsg('regMsg', data.message, 'success');
    document.getElementById('regName').value = '';
    loadEmployees();
  } else {
    showMsg('regMsg', data.message, 'error');
  }
}

async function loadEmployees() {
  const resp = await fetch('/api/admin/employees?password=' + encodeURIComponent(adminPwd));
  const data = await resp.json();
  const el = document.getElementById('employeeList');
  if (!data.employees || !data.employees.length) {
    el.innerHTML = '<div style="color:var(--muted);text-align:center;padding:1.5rem">Нет зарегистрированных сотрудников</div>';
    return;
  }
  el.innerHTML = data.employees.map(name => `
    <div class="emp-row">
      <div class="emp-name">${name}</div>
      <button class="btn btn-danger" onclick="deleteEmployee('${name}')">Удалить</button>
    </div>
  `).join('');
}

async function deleteEmployee(name) {
  if (!confirm('Удалить сотрудника ' + name + '?')) return;
  const resp = await fetch('/api/admin/delete-employee', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ name, password: adminPwd })
  });
  const data = await resp.json();
  if (data.success) loadEmployees();
  else alert(data.message);
}
</script>
</body>
</html>
"""


def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_faces():
    if not os.path.exists(FACES_FILE):
        return {}
    try:
        with open(FACES_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def save_faces(faces):
    with open(FACES_FILE, 'wb') as f:
        pickle.dump(faces, f)

def save_temp_image(image_base64):
    from PIL import Image
    raw = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    tmp_path = os.path.join(os.path.dirname(DATA_FILE) or '.', '_tmp_face.jpg')
    img.save(tmp_path, 'JPEG', quality=90)
    return tmp_path

def get_embedding(image_base64):
    logger.info('Face embedding requested: payload_size=%s', len(image_base64 or ''))
    tmp = save_temp_image(image_base64)
    try:
        result = DeepFace.represent(
            img_path=tmp,
            model_name='Facenet',
            enforce_detection=True,
            detector_backend='opencv'
        )
        if not result:
            return None, "Лицо не обнаружено"
        return np.array(result[0]['embedding']), None
    except Exception as e:
        msg = str(e)
        logger.exception('DeepFace.represent failed')
        if 'Face could not be detected' in msg or 'cannot detect' in msg.lower():
            return None, "Лицо не обнаружено. Встаньте прямо, обеспечьте хорошее освещение."
        return None, f"Ошибка: {msg}"
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))

def register_face(name, image_base64):
    logger.info('Register face called: name=%s', name)
    if not FACE_RECOGNITION_AVAILABLE:
        reason = f" Причина: {DEEPFACE_IMPORT_ERROR}" if DEEPFACE_IMPORT_ERROR else ''
        logger.error('Register face unavailable: %s', reason)
        return False, f"deepface недоступен.{reason}"
    embedding, err = get_embedding(image_base64)
    if embedding is None:
        return False, err or "Лицо не найдено на фото. Встаньте ближе к камере."
    faces = load_faces()
    faces[name.lower()] = {
        'display_name': name,
        'encoding': embedding.tolist()
    }
    save_faces(faces)
    return True, f'Сотрудник "{name}" зарегистрирован'

def identify_face(image_base64):
    logger.info('Identify face called: payload_size=%s', len(image_base64 or ''))
    if not FACE_RECOGNITION_AVAILABLE:
        reason = f" Причина: {DEEPFACE_IMPORT_ERROR}" if DEEPFACE_IMPORT_ERROR else ''
        logger.error('Identify face unavailable: %s', reason)
        return False, f"deepface недоступен.{reason}", None
    embedding, err = get_embedding(image_base64)
    if embedding is None:
        return False, err or "Лицо не обнаружено.", None
    faces = load_faces()
    if not faces:
        return False, "В системе нет зарегистрированных сотрудников. Обратитесь к администратору.", None

    best_name = None
    best_dist = float('inf')
    for key, val in faces.items():
        dist = cosine_distance(embedding, np.array(val['encoding']))
        if dist < best_dist:
            best_dist = dist
            best_name = key

    if best_dist <= FACE_DISTANCE_THRESHOLD:
        display = faces[best_name]['display_name']
        return True, f"Распознан: {display}", display
    return False, "Лицо не совпадает с базой. Вы зарегистрированы в системе?", None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info('HTTP %s - %s', self.address_string(), fmt % args)

    def _json(self, status, data):
        logger.info('Respond JSON: status=%s path=%s', status, self.path)
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content, status=200):
        logger.info('Respond HTML: status=%s path=%s', status, self.path)
        body = content.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        logger.info('Read JSON body: path=%s length=%s', self.path, length)
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        p = urlparse(self.path).path
        logger.info('Incoming GET: path=%s query=%s', p, urlparse(self.path).query)

        try:
            if p == '/':
                self._html(MAIN_HTML)
            elif p == '/admin':
                self._html(ADMIN_HTML)
            elif p == '/api/status':
                today = now_atyrau().strftime('%Y-%m-%d')
                data = load_data()
                self._json(200, data.get(today, []))
            elif p == '/api/time_left':
                now = now_atyrau()
                end_dt = datetime.combine(now.date(), WORK_END, tzinfo=ATYRAU_TZ)
                if now > end_dt:
                    self._json(200, {'time_left': 'Рабочий день окончен', 'finished': True})
                else:
                    diff = end_dt - now
                    h, rem = divmod(int(diff.total_seconds()), 3600)
                    m, s = divmod(rem, 60)
                    self._json(200, {'time_left': f'{h:02d}:{m:02d}:{s:02d}', 'finished': False})
            elif p.startswith('/api/admin/employees'):
                qs = parse_qs(urlparse(self.path).query)
                pw = qs.get('password', [''])[0]
                if pw != ADMIN_PASSWORD:
                    self._json(403, {'error': 'Неверный пароль'})
                    return
                faces = load_faces()
                names = [faces[k]['display_name'] for k in faces]
                self._json(200, {'employees': names})
            elif p == '/api/debug/runtime':
                self._json(200, runtime_diagnostics())
            else:
                self._html('<h1>404</h1>', 404)
        except Exception:
            logger.error('Unhandled GET error path=%s\n%s', p, traceback.format_exc())
            self._json(500, {'error': 'Внутренняя ошибка сервера. Смотрите логи Railway.'})

    def do_POST(self):
        p = urlparse(self.path).path
        logger.info('Incoming POST: path=%s', p)

        try:
            if p == '/api/identify-face':
                payload = self._read_json()
                image = payload.get('image', '')
                success, message, name = identify_face(image)
                if not success:
                    self._json(200, {'success': False, 'message': message})
                    return
                today = now_atyrau().strftime('%Y-%m-%d')
                data = load_data()
                already = any(e['name'].lower() == name.lower() for e in data.get(today, []))
                self._json(200, {'success': True, 'name': name, 'already_checked_in': already})

            elif p == '/api/face-checkin':
                payload = self._read_json()
                name = payload.get('name', '').strip()
                status_type = payload.get('status', 'present')
                if not name:
                    self._json(400, {'error': 'Имя не передано'})
                    return
                now = now_atyrau()
                today = now.strftime('%Y-%m-%d')
                data = load_data()
                data.setdefault(today, [])
                for entry in data[today]:
                    if entry['name'].lower() == name.lower():
                        self._json(400, {'error': 'Уже отмечен сегодня'})
                        return
                is_late = False
                if status_type == 'present':
                    start_dt = datetime.combine(now.date(), WORK_START, tzinfo=ATYRAU_TZ)
                    is_late = now > start_dt
                    late_min = int((now - start_dt).total_seconds() / 60)
                    status_text = f'Опоздал на {late_min} мин.' if is_late else 'Вовремя'
                elif status_type == 'absent':
                    status_text = 'Не придет'
                else:
                    status_text = 'В отпуске'
                entry = {
                    'name': name,
                    'time': now.strftime('%H:%M'),
                    'status': status_text,
                    'type': status_type,
                    'is_late': is_late,
                    'verified': True
                }
                data[today].append(entry)
                save_data(data)
                self._json(200, entry)

            elif p == '/api/admin/auth':
                payload = self._read_json()
                ok = payload.get('password') == ADMIN_PASSWORD
                self._json(200, {'success': ok})

            elif p == '/api/admin/register-face':
                payload = self._read_json()
                if payload.get('password') != ADMIN_PASSWORD:
                    self._json(403, {'success': False, 'message': 'Неверный пароль'})
                    return
                name = payload.get('name', '').strip()
                image = payload.get('image', '')
                if not name:
                    self._json(400, {'success': False, 'message': 'Введите имя'})
                    return
                ok, msg = register_face(name, image)
                self._json(200, {'success': ok, 'message': msg})

            elif p == '/api/admin/delete-employee':
                payload = self._read_json()
                if payload.get('password') != ADMIN_PASSWORD:
                    self._json(403, {'success': False, 'message': 'Неверный пароль'})
                    return
                name = payload.get('name', '').strip().lower()
                faces = load_faces()
                deleted = False
                for key in list(faces.keys()):
                    if key.lower() == name:
                        del faces[key]
                        deleted = True
                        break
                if deleted:
                    save_faces(faces)
                    self._json(200, {'success': True})
                else:
                    self._json(404, {'success': False, 'message': 'Сотрудник не найден'})

            else:
                self._html('<h1>404</h1>', 404)
        except Exception:
            logger.error('Unhandled POST error path=%s\n%s', p, traceback.format_exc())
            self._json(500, {'error': 'Внутренняя ошибка сервера. Смотрите логи Railway.'})


if __name__ == '__main__':
    logger.info('Web app starting on port=%s', PORT)
    logger.info('DeepFace available=%s', FACE_RECOGNITION_AVAILABLE)
    logger.info('Runtime diagnostics: %s', json.dumps(runtime_diagnostics(), ensure_ascii=False))
    logger.info('Откройте браузер: http://localhost:%s', PORT)
    logger.info('Админ-панель:     http://localhost:%s/admin  (пароль: %s)', PORT, ADMIN_PASSWORD)

    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
