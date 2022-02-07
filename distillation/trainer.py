import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow_datasets as tfds
from components import *

optimizer = tf.keras.optimizers.Adam(0.001)
teacher = tf.keras.models.load_model('Actual_teacher')
student = tf.keras.models.load_model('Actual_student')
print("Teacher: ")
print(teacher.summary())
print("Student: ")
print(student.summary())

ds_train, ds_test = preprocessor()
EPOCHS = 5
for i in range(0, EPOCHS):
    for step, (x_batch_train, y_batch_train) in enumerate(ds_train):

        with tf.GradientTape() as tape:
            teacher_out = teacher(x_batch_train)
            student_out = student(x_batch_train)
            l1_value = tf.keras.losses.mean_squared_error(
                teacher_out, student_out)
            L = tf.keras.losses.SparseCategoricalCrossentropy()
            loss_value = 10**2*l1_value+L(y_batch_train, student_out)
        grads = tape.gradient(loss_value, student.trainable_weights)
        # print(grads)
        optimizer.apply_gradients(zip(grads, student.trainable_weights))
        if(step % 20 == 0):
            print("Step number "+str(step)+" in epoch number "+str(i+1))
    print("Epoch number "+str(i+1)+"done")

student.save("Trained_student")
