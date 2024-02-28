import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.fashion_mnist

# Load data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize data
training_images, test_images = training_images / 255.0, test_images / 255.0

# Define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer (2)
    keras.layers.Dense(10, activation=tf.nn.softmax)  # output layer (3)
])

# Compile model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(training_images, training_labels, epochs=5)

# Evaluate model
model.evaluate(test_images, test_labels)

# Make predictions
classifications = model.predict(test_images)
print(classifications[0])
print('\n')

# Print label
print(test_labels[0])
