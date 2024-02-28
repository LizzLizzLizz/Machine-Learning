import os
import zipfile
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dropout

# DATA PREPARATION
# ngambil dataset
!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

# ekstrak datasetnya
local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# direktori sementara
base_dir = '/tmp/rockpaperscissors'

# bikin direktori buat data latih sama data uji
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# bikin direktori buat setiap kelas
rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

# bagi dataset jadi data latih sama data uji dengan perbandingan 40:60
rock_train, rock_val = train_test_split(os.listdir(rock_dir), test_size=0.4)
paper_train, paper_val = train_test_split(os.listdir(paper_dir), test_size=0.4)
scissors_train, scissors_val = train_test_split(os.listdir(scissors_dir), test_size=0.4)

# bikin direktori baru di dalem direktori data latih sama data uji
train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

val_rock_dir = os.path.join(validation_dir, 'rock')
val_paper_dir = os.path.join(validation_dir, 'paper')
val_scissors_dir = os.path.join(validation_dir, 'scissors')

os.makedirs(train_rock_dir, exist_ok=True)
os.makedirs(train_paper_dir, exist_ok=True)
os.makedirs(train_scissors_dir, exist_ok=True)

os.makedirs(val_rock_dir, exist_ok=True)
os.makedirs(val_paper_dir, exist_ok=True)
os.makedirs(val_scissors_dir, exist_ok=True)

# ngecopy dataset ke direktori baru
for file in rock_train:
    shutil.copy(os.path.join(rock_dir, file), os.path.join(train_rock_dir, file))
for file in paper_train:
    shutil.copy(os.path.join(paper_dir, file), os.path.join(train_paper_dir, file))
for file in scissors_train:
    shutil.copy(os.path.join(scissors_dir, file), os.path.join(train_scissors_dir, file))

for file in rock_val:
    shutil.copy(os.path.join(rock_dir, file), os.path.join(val_rock_dir, file))
for file in paper_val:
    shutil.copy(os.path.join(paper_dir, file), os.path.join(val_paper_dir, file))
for file in scissors_val:
    shutil.copy(os.path.join(scissors_dir, file), os.path.join(val_scissors_dir, file))

# BUILDING MODEL
# bikin image augmentation dengan berbagai teknik
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# bikin generator buat data latih sama data uji
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='sparse'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='sparse'
)

# arsitektur model menggunakan sequential dan menggunakan banyak hidden layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# ngecompile model menggunakan adam optimizer dan menggunakan sparse categorical crossentropy sebagai loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# callback buat berhentiin training kalo udah dapet akurasi 96%
class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] >= 0.96:
            self.model.stop_training = True

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# ngelatih model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=25,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[AccuracyCallback(), early_stopping, lr_scheduler_callback],
    verbose=2
)

# DEPLOYMENT
# nampilin grafik akurasi dan loss
print(train_generator.class_indices)

# upload foto buat testing
uploaded = files.upload()

# hasil testing
for fn in uploaded.keys():
    path = fn
    img = image.load_img(path, target_size=(150, 150))

    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    probabilities = model.predict(x, batch_size=1)[0]
    class_labels = ['paper', 'rock', 'scissors']
    predicted_class = class_labels[np.argmax(probabilities)]
    highest_probability = np.max(probabilities)

    print(fn)
    print(f'Probabilitas tiap kelas : {probabilities}')
    print(f'Prediksi kelas : {predicted_class} dengan probabilitas : {highest_probability}')
