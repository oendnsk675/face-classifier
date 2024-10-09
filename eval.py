import sys
sys.path.append("src")

import torch
import cv2  # Untuk membuka kamera
import torchvision.transforms as transforms
from PIL import Image
from face_classify.trainer.trainer import Trainer

# Menambahkan path


# Load model dari checkpoint
checkpoint_path = "logs/csv_logger/version_1/checkpoints/epoch=9-step=3870.ckpt"
model = Trainer.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
model.eval()  # Set model ke mode evaluasi
model.freeze()  # Freeze model untuk inference

# Definisikan transformasi gambar
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cpu"
model = model.to(device)

class_names = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Fungsi untuk prediksi
def predict_frame(frame, model):
    # Konversi frame OpenCV (BGR) ke format PIL (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Transformasi gambar
    img = transform(img).unsqueeze(0)  # Tambah dimensi batch
    
    # Prediksi
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
    
    # Ambil prediksi kelas
    pred_class = torch.argmax(output, dim=1).item()
    pred_class_name = class_names[pred_class]
    
    return pred_class_name

# Buka kamera menggunakan OpenCV
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

if not cap.isOpened():
    print("Error: Kamera tidak tersedia.")
    exit()

print("Tekan 'q' untuk keluar")

# Loop untuk menangkap frame dari kamera secara real-time
while True:
    ret, frame = cap.read()  # Baca frame dari kamera
    if not ret:
        print("Gagal menangkap frame")
        break

    # Lakukan prediksi untuk frame saat ini
    pred_class_name = predict_frame(frame, model)

    # Tampilkan hasil prediksi di frame
    cv2.putText(frame, f"Predicted: {pred_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan frame dengan OpenCV
    cv2.imshow('Real-time Emotion Prediction', frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan capture dan tutup jendela
cap.release()
cv2.destroyAllWindows()
