#dataları yan yana görmek için çalıştırabiliriz

from PIL import Image
import matplotlib.pyplot as plt

def rotate_image(image_path, degrees):
    # Orijinal görüntüyü aç
    original_image = Image.open(image_path)
    
    # Orijinal görüntüyü 80x80 boyutuna yeniden boyutlandır
    original_image = original_image.resize((80, 80))
    
    # Dereceye göre görüntüyü döndür
    rotated_image = original_image.rotate(degrees, expand=True)
    
    # Döndürülmüş görüntüyü 80x80 boyutuna yeniden boyutlandır
    rotated_image = rotated_image.resize((80, 80))
    
    return rotated_image

# Orijinal görsel yolu
image_path = "dataset/train/Salatalık/0001.jpg"

# Dereceler listesi: 90, 180 ve 270
degrees_list = [90, 180, 270]

# Orijinal görseli ve döndürülmüş görselleri içerecek bir figür oluştur
fig, axes = plt.subplots(1, len(degrees_list) + 1)

# Orijinal görseli figürün ilk alt grafiğine ekle
axes[0].imshow(rotate_image(image_path, 0))
axes[0].axis("off")
axes[0].set_title("Orijinal")

# Döndürülmüş görselleri figürün diğer alt grafiklerine ekle
for i, degrees in enumerate(degrees_list):
    rotated_image = rotate_image(image_path, degrees)
    axes[i+1].imshow(rotated_image)
    axes[i+1].axis("off")
    axes[i+1].set_title(f"{degrees} Derece")
    
# Grafikleri göster
plt.show()
