import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Modeli yükleyin
model_path = 'model(5000-5000)(SGD)(20epoch).h5'
model = tf.keras.models.load_model(model_path)

# Test görseli
img_path = 'deneme.JPG'
categories = ['10kurus', '1tl', '25kurus', '50kurus', '5kurus']

# Görseli yükleyin ve işleyin
img = cv2.imread(img_path)
output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Hough Circle Transform ile daireleri tespit et
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=600, param1=100, param2=30, minRadius=200, maxRadius=600)

# Her paranın sayısını ve toplam değerini hesaplama
coin_counts = {0.1: 0, 1.0: 0, 0.25: 0, 0.5: 0, 0.05: 0}
value_map = {'10kurus': 0.1, '1tl': 1.0, '25kurus': 0.25, '50kurus': 0.5, '5kurus': 0.05}

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Daire içine alarak parayı kes
        margin = 30  # Kenarlardan ekleyeceğimiz piksel sayısı
        x1 = max(0, x - r - margin)
        y1 = max(0, y - r - margin)
        x2 = min(img.shape[1], x + r + margin)
        y2 = min(img.shape[0], y + r + margin)
        coin = img[y1:y2, x1:x2]  # Daire içindeki parayı kenarlarıyla birlikte keser
        if coin.shape[0] == 0 or coin.shape[1] == 0:
            continue

        # Önce 320x320 boyutunda yeniden boyutlandır
        coin_resized = cv2.resize(coin, (512, 512))  # Parayı (512, 512) boyutunda yeniden boyutlandırır

        # Görüntüyü göster
        cv2.imshow('Algilanan Para', coin_resized)
        cv2.waitKey(0)  # Her bir görüntü için beklemek, bir tuşa basana kadar geçer

        if cv2.waitKey(0) == ord('q'):
            break;

        # Model için ön işleme (normalize eder)
        coin_resized = image.img_to_array(coin_resized)
        coin_resized = np.expand_dims(coin_resized, axis=0) / 255.0

        # Tahmin yapma ve çıktıları yazdırma
        predictions = model.predict(coin_resized)
        print(f"Predictions: {predictions}")  # Tahminlerin olasılık dağılımını yazdır
        predicted_class = np.argmax(predictions[0])
        predicted_label = categories[predicted_class]

        # Paranın değerini belirle ve güncelle
        coin_value = value_map[predicted_label]
        coin_counts[coin_value] += 1

        # Görsel üzerine yazdırma
        cv2.putText(output, f"{predicted_label}", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),2)  # Tahmini yazar
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Daireyi çizer

total_value = sum(value * count for value, count in coin_counts.items())
total_value = round(total_value, 2)

# Toplam değeri görsel üzerine yaz
cv2.putText(output, f"Toplam: {total_value} TL", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

resized_output = cv2.resize(output, (900, 900))

# Sonuç görselini göster
cv2.imshow("Coins Detected", resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()