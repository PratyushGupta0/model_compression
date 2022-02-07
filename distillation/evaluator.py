import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow_datasets as tfds
from components import *
ds_train,ds_test=preprocessor()
student=tf.keras.models.load_model('Trained_student')

def evaluator_model(student):
    inputs=student.input
    x=student.layers[-2].output
    outputs=tf.nn.softmax(x)
    return tf.keras.Model(inputs,outputs)

evaluator=evaluator_model(student)
evaluator.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
evaluator.evaluate(ds_test)