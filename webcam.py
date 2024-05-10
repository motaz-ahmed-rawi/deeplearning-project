import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modelinizi yükleyin
model = load_model('lab3.h5')  # Modelinizi kaydettiğiniz dosya yolunu buraya girin

# Preprocess the image
def preprocessing(img):
    img = img / 255.0
    img = cv2.resize(img, (48, 48))
    return img.reshape(-1, 48, 48, 1)  # Reshape to match input shape

# Kamerayı açın
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Yüz tespiti yapın
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Tespit edilen her yüz için işlem yapın
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]  # Yüz bölgesini alın
        processed_img = preprocessing(face_img)  # Veriyi ön işleyin
        prediction = model.predict(processed_img)  # Modelden tahmin alın
        
        # Tahmin sonuçlarını etiketleyin ve görüntüleyin
        if prediction[0][0] > prediction[0][1]:
            label = "Sad"
        else:
            label = "Happy"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Sonuçları görüntüleyin
    cv2.imshow('Face Recognition', frame)
    
    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapatın
cap.release()
cv2.destroyAllWindows()
