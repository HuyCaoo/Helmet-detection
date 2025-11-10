from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Cấu hình
# -----------------------------
MODEL_PATH = "helmetYoloV8_100epochs.pt"   # đường dẫn model YOLOv8
IMAGE_PATH = "test2.jpg"       # đường dẫn ảnh cần test
CONFIDENCE = 0.5              # confidence threshold
SHOW_IMAGE = True             # True = hiển thị ảnh, False = không hiển thị

# -----------------------------
# Kiểm tra file tồn tại
# -----------------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Load ảnh
# -----------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Cannot read image: {IMAGE_PATH}")

# -----------------------------
# Detect
# -----------------------------
results = model(img, conf=CONFIDENCE)

# -----------------------------
# Vẽ kết quả lên ảnh
# -----------------------------
annotated_img = results[0].plot()

# -----------------------------
# Hiển thị ảnh với OpenCV
# -----------------------------
if SHOW_IMAGE:
    cv2.imshow("Detection Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# Lưu kết quả ra file
# -----------------------------
output_path = "result.jpg"
cv2.imwrite(output_path, annotated_img)
print(f"Result saved to {output_path}")

# -----------------------------
# In thông tin detection ra console
# -----------------------------
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()   # bounding boxes
    scores = r.boxes.conf.cpu().numpy()  # confidence
    classes = r.boxes.cls.cpu().numpy()  # class id
    for i, box in enumerate(boxes):
        print(f"Class: {int(classes[i])}, Confidence: {scores[i]:.2f}, Box: {box}")
