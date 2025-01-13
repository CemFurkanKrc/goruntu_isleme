import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam , SGD
import tensorflow as tf

# Veri Hazırlığı
train_dir = 'NewDataset5000/train'
validation_dir = 'NewDataset5000/validation'
categories = ['10kurus', '1tl', '25kurus', '50kurus', '5kurus']

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# tf.data.Dataset API'si ile veri setini oluşturma
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(categories)), dtype=tf.float32)
    )
)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(categories)), dtype=tf.float32)
    )
)

# Model Tanımlaması
base_model = VGG16(input_shape=(512, 512, 3), include_top=False, weights='imagenet')
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Modeli kaydet
model.save('model(5000-5000)(SGD)(20epoch).h5')