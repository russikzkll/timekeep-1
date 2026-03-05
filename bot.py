import json
import os
import pickle
import math
import io
import tempfile
import numpy as np
from datetime import datetime, time, timezone, timedelta
from PIL import Image

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)
from telegram.error import Conflict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BOT_TOKEN = os.getenv("BOT_TOKEN", "")

OFFICE_LAT = float(os.getenv("OFFICE_LAT", "47.11150731320942"))
OFFICE_LON = float(os.getenv("OFFICE_LON", "51.92043675195111"))
OFFICE_RADIUS_M = int(os.getenv("OFFICE_RADIUS_M", "100"))

FACES_FILE = "faces_data.pkl"
ATTENDANCE_FILE = "attendance.json"

WORK_START = time(14, 0)
WORK_END = time(17, 30)

FACE_THRESHOLD = 0.40


ATYRAU_TZ = timezone(timedelta(hours=5), "Asia/Atyrau")


def now_atyrau() -> datetime:
    return datetime.now(ATYRAU_TZ)


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

DEEPFACE_IMPORT_ERROR = ""
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception as e:
    DEEPFACE_OK = False
    DEEPFACE_IMPORT_ERROR = str(e)

SELECT_NAME, VERIFY_FACE, SEND_LOCATION, SELECT_STATUS = range(4)


def load_faces():
    if not os.path.exists(FACES_FILE):
        return {}
    with open(FACES_FILE, "rb") as f:
        return pickle.load(f)


def load_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return {}
    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_attendance(data):
    with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def already_checked_in(name: str) -> bool:
    today = now_atyrau().strftime("%Y-%m-%d")
    data = load_attendance()
    return any(e["name"].lower() == name.lower() for e in data.get(today, []))


def record_attendance(name: str, status_type: str) -> str:
    now = now_atyrau()
    today = now.strftime("%Y-%m-%d")
    data = load_attendance()
    data.setdefault(today, [])

    is_late = False
    if status_type == "present":
        start_dt = datetime.combine(now.date(), WORK_START, tzinfo=ATYRAU_TZ)
        is_late = now > start_dt
        late_min = int((now - start_dt).total_seconds() / 60)
        status_text = f"Опоздал на {late_min} мин." if is_late else "Вовремя"
    elif status_type == "absent":
        status_text = "Не придет"
    else:
        status_text = "В отпуске"

    data[today].append({
        "name": name,
        "time": now.strftime("%H:%M"),
        "status": status_text,
        "type": status_type,
        "is_late": is_late,
        "verified": True,
        "source": "telegram",
    })
    save_attendance(data)
    return status_text


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def verify_face_bytes(image_bytes: bytes, expected_name: str) -> tuple[bool, str]:
    if not DEEPFACE_OK:
        reason = f" Причина: {DEEPFACE_IMPORT_ERROR}" if DEEPFACE_IMPORT_ERROR else ""
        return False, f"DeepFace недоступен.{reason}"

    faces = load_faces()
    key = expected_name.lower()
    if key not in faces:
        return False, "Сотрудник не найден в базе."

    stored = np.array(faces[key]["encoding"])

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(tmp_path, "JPEG")

    try:
        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv",
        )
        if not result:
            return False, "Лицо не обнаружено на фото."
        embedding = np.array(result[0]["embedding"])
        dist = cosine_distance(embedding, stored)
        if dist <= FACE_THRESHOLD:
            return True, "ok"
        return False, "Лицо не совпадает."
    except Exception as e:
        msg = str(e)
        if "Face could not be detected" in msg or "cannot detect" in msg.lower():
            return False, "Лицо не обнаружено. Сделайте четкое фото при хорошем освещении."
        return False, f"Ошибка распознавания: {msg}"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    faces = load_faces()
    if not faces:
        await update.message.reply_text(
            "База сотрудников пуста. Обратитесь к администратору.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    names = [faces[k]["display_name"] for k in faces]
    keyboard = [[name] for name in names]

    await update.message.reply_text(
        "Выберите своё имя из списка:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return SELECT_NAME


async def handle_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chosen = update.message.text.strip()
    faces = load_faces()
    display_names = {faces[k]["display_name"].lower(): faces[k]["display_name"] for k in faces}

    if chosen.lower() not in display_names:
        await update.message.reply_text(
            "Вас нет в базе. Обратитесь к администратору.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    real_name = display_names[chosen.lower()]

    if already_checked_in(real_name):
        await update.message.reply_text(
            f"Вы уже отмечены сегодня.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    context.user_data["name"] = real_name

    await update.message.reply_text(
        f"Отправьте своё фото для подтверждения личности.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return VERIFY_FACE


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = context.user_data.get("name")

    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_bytes = await file.download_as_bytearray()

    await update.message.reply_text("Проверяю...")

    ok, msg = verify_face_bytes(bytes(image_bytes), name)

    if not ok:
        await update.message.reply_text(
            f"Верификация не пройдена: {msg}\n\nНачните заново — /start",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    location_btn = KeyboardButton("Отправить геолокацию", request_location=True)
    await update.message.reply_text(
        f"Личность подтверждена. Теперь отправьте геолокацию.",
        reply_markup=ReplyKeyboardMarkup([[location_btn]], one_time_keyboard=True, resize_keyboard=True),
    )
    return SEND_LOCATION


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = context.user_data.get("name")
    loc = update.message.location

    distance = haversine_distance(loc.latitude, loc.longitude, OFFICE_LAT, OFFICE_LON)
    dist_int = int(distance)

    if distance > OFFICE_RADIUS_M:
        await update.message.reply_text(
            f"Вы находитесь слишком далеко от офиса ({dist_int} м).\n"
            f"Допустимый радиус: {OFFICE_RADIUS_M} м.\n\n"
            f"Отметка невозможна.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    keyboard = [["Я пришёл"], ["Не приду"], ["В отпуске"]]
    await update.message.reply_text(
        f"Вы в офисе ({dist_int} м от точки). Выберите статус:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return SELECT_STATUS


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = context.user_data.get("name")
    text = update.message.text.strip()

    mapping = {
        "Я пришёл": "present",
        "Не приду": "absent",
        "В отпуске": "vacation",
    }

    if text not in mapping:
        await update.message.reply_text(
            "Выберите один из вариантов.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return SELECT_STATUS

    status_type = mapping[text]
    status_text = record_attendance(name, status_type)

    await update.message.reply_text(
        f"Готово, {name}.\nСтатус: {status_text}",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отменено.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, Conflict):
        print(
            "Ошибка Telegram Conflict: одновременно запущено несколько экземпляров бота "
            "с одним BOT_TOKEN. Оставьте только один инстанс/реплику."
        )
        return
    print(f"Необработанная ошибка: {err}")


def main():
    if not BOT_TOKEN:
        print("Ошибка: BOT_TOKEN не задан в .env")
        return
    print("Telegram бот запущен.")

    app = Application.builder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            SELECT_NAME:   [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_name)],
            VERIFY_FACE:   [MessageHandler(filters.PHOTO, handle_photo)],
            SEND_LOCATION: [MessageHandler(filters.LOCATION, handle_location)],
            SELECT_STATUS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_status)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(conv)
    app.add_error_handler(error_handler)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()