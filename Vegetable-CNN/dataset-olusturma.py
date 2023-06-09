#veri genişletme işlemi ile dataset oluşturuldu (çalıştırmaya bir daha gerek yok)

import os
from PIL import Image

def rotate_images_in_folder(folder_path):
    # Klasördeki tüm dosyaları dolaş
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            
            # Orijinal görüntüyü aç
            original_image = Image.open(image_path)
            
            # Görüntüyü 90, 180 ve 270 derece döndür
            for degrees in [90, 180, 270]:
                rotated_image = original_image.rotate(degrees, expand=True)
                
                # Döndürülmüş görüntüyü kaydet
                rotated_image_path = os.path.join(folder_path, f"{filename}_{degrees}.jpg")
                rotated_image.save(rotated_image_path)
                
                print(f"{filename} - {degrees} derece döndürüldü. Kaydedildi: {rotated_image_path}")


folder_path = "dataset/test/Karnabahar/"  # Klasörün yolunu belirtin
rotate_images_in_folder(folder_path)
