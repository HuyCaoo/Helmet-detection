from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Cấu hình
# -----------------------------
MODEL_PATH = "helmetYoloV8_100epochs.pt"   # đường dẫn model YOLOv8
VIDEO_PATH = "videotest1.mp4"  # đường dẫn video, hoặc 0 để dùng webcam
CONFIDENCE = 0.5             # confidence threshold
SHOW_VIDEO = True            # True = hiển thị video, False = chỉ lưu file
OUTPUT_VIDEO = "result_video.mp4"  # đường dẫn lưu video kết quả

# -----------------------------
# Kiểm tra file tồn tại (nếu không dùng webcam)
# -----------------------------
if VIDEO_PATH != "0" and not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Mở video / webcam
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != "0" else 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# -----------------------------
# Xử lý từng frame
# -----------------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect
    results = model(frame, conf=CONFIDENCE)
    annotated_frame = results[0].plot()

    # In thông tin detection
    has_helmet = 0
    no_helmet = 0
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        for c in classes:
            if int(c) == 1:  # giả sử class 1 = có mũ
                has_helmet += 1
            else:
                no_helmet += 1
    print(f"Frame {frame_count}: {has_helmet} có mũ, {no_helmet} không mũ")

    # Hiển thị
    if SHOW_VIDEO:
        cv2.imshow("Helmet Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # nhấn ESC để dừng
            break

    # Lưu frame vào video
    out.write(annotated_frame)

# -----------------------------
# Kết thúc
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video result saved to {OUTPUT_VIDEO}")
