from ultralytics import YOLO
import cv2

# Khởi tạo mô hình YOLO
model = YOLO('best.pt')

# Đường dẫn đến hình ảnh cần nhận diện
image_path = "test.jpg"

# Đọc hình ảnh
frame = cv2.imread(image_path)

# Phát hiện đối tượng
results = model(frame)

# Vẽ bounding box và hiển thị kết quả
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Lấy thông tin lớp (class) và độ tin cậy (confidence)
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Lấy tên lớp từ model.names
        class_name = model.names[class_id]

        # Hiển thị tên lớp và độ tin cậy
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Hiển thị kết quả trên cửa sổ
cv2.imshow('Image Detection', frame)

# Chờ người dùng nhấn phím để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()