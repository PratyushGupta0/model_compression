import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow_datasets as tfds
from components import *
model = tf.keras.models.load_model('teacher_model')
print(model.summary())
print(model.layers)

pre_softmaxer=pre_softmaxer(model)
student=student_maker(10.0)
student.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
student.build((None,28,28,1))
print(student.summary())

teacher=teacher(pre_softmaxer,10.0)
teacher.build((None,28,28,1))
print(teacher.summary())
teacher.save('Actual_teacher')
student.save("Actual_student")

