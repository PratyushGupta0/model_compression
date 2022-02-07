import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow_datasets as tfds
from components import *
print(tf.test.is_gpu_available())
ds_train, ds_test = preprocessor()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1280, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear'),
    tf.keras.layers.Softmax()
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test,
    verbose=False
)
print(model.summary())
model.evaluate(ds_test)
model.save('teacher_model')
