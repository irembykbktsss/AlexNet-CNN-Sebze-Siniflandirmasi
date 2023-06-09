import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.optimizers import Adam


train_path = "dataset/train"
validation_path = "dataset/validation"
test_path = "dataset/test"

image_categories = os.listdir('dataset/train')

#kategori bazlı verileri görselleştirme işlemi 
def plot_images(image_categories):
    # figure çiz
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        # i. kategori için resim yükle
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = tf.keras.preprocessing.image.load_img(first_image_path)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)/255.0

        # Subplot oluşturun ve görüntüleri çizin
        plt.subplot(1, 5, i+1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')
        
    plt.show()

plot_images(image_categories)

# Image Data Generator oluşturma (train, validation and test)

# 1. Train dataset
train_gen = ImageDataGenerator(rescale = 1.0/255.0) # normalize işlemi
train_image_generator = train_gen.flow_from_directory(
                                            train_path,
                                            target_size=(227, 227),
                                            batch_size=32,
                                            class_mode='categorical')

# 2. Validation dataset
val_gen = ImageDataGenerator(rescale = 1.0/255.0) # normalize işlemi
val_image_generator = train_gen.flow_from_directory(
                                            validation_path,
                                            target_size=(227, 227),
                                            batch_size=32,
                                            class_mode='categorical')

# 3. Test dataset
test_gen = ImageDataGenerator(rescale = 1.0/255.0) # normalize işlemi
test_image_generator = train_gen.flow_from_directory(
                                            test_path,
                                            target_size=(227, 227),
                                            batch_size=32,
                                            class_mode='categorical')

# Sınıf etiketlerini oluşturma işlemi
class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
print(class_map)


# Alexnet CNN model oluşturma işlemi
model = Sequential() 

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation='softmax')
])


# Model derleme ve eğitme işlemi
early_stopping = keras.callbacks.EarlyStopping(patience=5) # Set up callbacks

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics='accuracy')
hist = model.fit(train_image_generator, 
                 epochs=10,   #eğitim turu
                 verbose=1,   #eğitim sürecinin ayrıntılarının ekranda görüntülenmesini sağlar
                 validation_data=val_image_generator, 
                 steps_per_epoch = 20000//32,   #eğitim veri kümesindeki görüntülerin sayısı/her adımda 32 görüntü işleniyor
                 validation_steps = 1000//32,   #validation veri kümesindeki görüntülerin sayısı/her adımda 32 görüntü işleniyor
                 callbacks=early_stopping)



# Eğitim sonrası hata ve doğruluk oranlarını grafik olarak çizme işlemi
h = hist.history
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(h['loss'], c='red', label='Training Loss')
plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
plt.xlabel("Number of Epochs")
plt.legend(loc='best')
plt.show()


# Test verisi üzerinden doğruluk tahmin işlemi
model.evaluate(test_image_generator)

# test görsel yolu
test_image_path = 'dataset/test/Salatalık/1009.jpg_180.jpg'

def generate_predictions(test_image_path, actual_label):
    
    # Görseli yükle ve önişleme yap
    test_img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(227, 227))

    test_img_arr =  tf.keras.preprocessing.image.img_to_array(test_img)/255.0
    test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

    # Tahmin işlemini yap ve sonucu figure olarak görselleştir
    predicted_label = np.argmax(model.predict(test_img_input))
    predicted_vegetable = class_map[predicted_label]
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img_arr)
    plt.title("Predicted Label: {}, Actual Label: {}".format(predicted_vegetable, actual_label))
    plt.grid()
    plt.axis('off')
    plt.show()


generate_predictions(test_image_path, actual_label='Salatalık')

