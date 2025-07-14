import sys
import torch
import pathlib
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

# Fix PosixPath on Windows
pathlib.PosixPath = pathlib.WindowsPath

from ResNet import Resnet18
from Efficientnet_B0 import My_EfficientNet_B0

# Đường dẫn
repo_dir = r'D:/BTL_ThucTapCS/yolov5'
weights_path = r'D:/BTL_ThucTapCS/YoloV5_Finetuning2/results/yolov5/runs/train/my-yolov5s-fall-detection/weights/best.pt'

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load YOLOv5 model
model = torch.hub.load(
    repo_or_dir=repo_dir,
    model='custom',
    path=weights_path,
    source='local',
    force_reload=True,
    device=device
)
model.eval()

# Load checkpoint model
def load_model(model, path_checkpoint):
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

# Load các model phân loại
resnet_model = load_model(Resnet18(num_classes=2), r'D:/BTL_ThucTapCS/Model_Classification/no_KD/resnet/best_cnn.pt')
effi_model = load_model(My_EfficientNet_B0(num_classes=2), r'D:\BTL_ThucTapCS\Model_Classification\KD\effi\best_cnn.pt')

# Hàm preprocess ảnh
def preprocess_crop(crop_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

# Mở camera
video_path = r'D:\BTL_ThucTapCS\file_demo\demo.mp4'  # <-- Đường dẫn tới video
cap = cv2.VideoCapture(video_path)

# Tạo cửa sổ full màn hình
window_name = "YOLO + EfficientNet Classification"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # YOLOv5 detection
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        input_tensor = preprocess_crop(crop)

        # Classify với EfficientNet
        with torch.no_grad():
            outputs = effi_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf_score, pred_class = torch.max(probs, dim=1)

        label = f"{'Fall' if pred_class.item() == 0 else 'Normal'} ({conf_score.item():.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Tạo VideoWriter để lưu video mới
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # hoặc 'MP4V' cho mp4
# out = cv2.VideoWriter('output2.avi', fourcc, fps, (width, height))
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to PIL image
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # YOLOv5 detection
#     results = model(img)
#     detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls
#
#     for det in detections:
#         x1, y1, x2, y2, conf, cls = map(int, det[:6])
#         crop = frame[y1:y2, x1:x2]
#
#         if crop.size == 0:
#             continue
#
#         input_tensor = preprocess_crop(crop)
#
#         # Classify với EfficientNet
#         with torch.no_grad():
#             outputs = effi_model(input_tensor)
#             probs = torch.softmax(outputs, dim=1)
#             conf_score, pred_class = torch.max(probs, dim=1)
#
#         label = f"{'Fall' if pred_class.item() == 0 else 'Normal'} ({conf_score.item():.2f})"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x1, max(0, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#
#     # Hiển thị và ghi frame đã xử lý
#     cv2.imshow(window_name, frame)
#     out.write(frame)  # Ghi frame ra video
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng tài nguyên
# cap.release()
# out.release()
# cv2.destroyAllWindows()