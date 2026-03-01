import cv2
import time

# Загружаем каскады
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def is_face(roi_gray, x, y, w, h):
    aspect_ratio = w / h
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False
    roi_upper = roi_gray[0:int(h*0.6), 0:w]
    eyes = eye_cascade.detectMultiScale(roi_upper, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    return len(eyes) > 0

def detect_face(frame):
    gray = preprocess(frame)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if is_face(roi_gray, x, y, w, h):
            return (x, y, w, h)
    return None

def detect_smile(roi_gray):
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
    return len(smiles) > 0

# Подключаем камеру
cap = cv2.VideoCapture(0)

# Ждём лицо
print("Ожидание лица...")
bbox = None
while bbox is None:
    ret, frame = cap.read()
    if not ret:
        break
    bbox = detect_face(frame)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if bbox is None:
    print("Лицо не обнаружено.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
print("Трекинг начат. Нажмите 'q' для выхода.")

# Для подсчёта FPS
fps_counter = 0
fps_start_time = time.time()
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обновление трекера
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Анализ эмоций (улыбка) в области лица
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        smiling = detect_smile(roi_gray)
        emotion_text = "Smiling" if smiling else "Neutral"
        cv2.putText(frame, emotion_text, (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost, redetecting...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        new_bbox = detect_face(frame)
        if new_bbox is not None:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, new_bbox)
            bbox = new_bbox
            print("Лицо восстановлено")

    # Подсчёт FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face Tracking + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()