!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

# melakukan ekstraksi pada file zip
import zipfile, os

local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/rockpaperscissors'

# Membuat direktori untuk data training dan data validasi
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Membuat direktori untuk setiap kelas
rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

# Split data menjadi train dan validation set
from sklearn.model_selection import train_test_split

rock_train, rock_val = train_test_split(os.listdir(rock_dir), test_size=0.4)
paper_train, paper_val = train_test_split(os.listdir(paper_dir), test_size=0.4)
scissors_train, scissors_val = train_test_split(os.listdir(scissors_dir), test_size=0.4)

# Membuat direktori baru di dalam direktori data training dan data validation
train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

val_rock_dir = os.path.join(validation_dir, 'rock')
val_paper_dir = os.path.join(validation_dir, 'paper')
val_scissors_dir = os.path.join(validation_dir, 'scissors')

# Membuat direktori baru
os.makedirs(train_rock_dir, exist_ok=True)
os.makedirs(train_paper_dir, exist_ok=True)
os.makedirs(train_scissors_dir, exist_ok=True)

os.makedirs(val_rock_dir, exist_ok=True)
os.makedirs(val_paper_dir, exist_ok=True)
os.makedirs(val_scissors_dir, exist_ok=True)

# Menyalin file ke direktori baru
import shutil

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

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dropout

# Augmentasi gambar
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

# Membuat generator untuk data training dan data validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Membuat model sequential dengan beberapa hidden layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add another dropout layer
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile model dengan 'adam' optimizer dan loss function 'categorical_crossentropy'
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback untuk menghentikan pelatihan jika akurasi sudah mencapai 96%
class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] >= 0.96:
            self.model.stop_training = True

# Callback untuk early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Callback untuk learning rate scheduling
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Pelatihan model dengan model.fit
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[AccuracyCallback(), early_stopping, lr_scheduler_callback]
)

import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
 
uploaded = files.upload()
 
for fn in uploaded.keys():
    # predicting images
    path = fn
    img = image.load_img(path, target_size=(150,150))
 
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
 
    # Use model.predict to get the probability
    probability = model.predict(images, batch_size=10)[0][0]
    print(fn)
    if probability > 0.5:
        print(f'messy with probability {probability}')
    else:
        print(f'clean with probability {1 - probability}')
