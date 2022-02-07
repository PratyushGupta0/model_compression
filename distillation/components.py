import tensorflow as tf
import tensorflow_datasets as tfds

class modded_softmax(tf.keras.layers.Layer):
    def __init__(self, T):
        super((modded_softmax), self).__init__()
        self.T = T
            
    def call(self, inputs):
        x=tf.math.divide(inputs,self.T)
        return tf.nn.softmax(x)

def pre_softmaxer(model):
    inputs=model.input
    output=model.layers[-1].output
    return tf.keras.Model(inputs,output)

def student_maker(T):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16,kernel_size=3),
        tf.keras.layers.Conv2D(filters=16,kernel_size=5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(10,activation='linear'),
        modded_softmax(T),
    ])
def teacher(pre_softmaxer,T):
    inputs=pre_softmaxer.input
    x=pre_softmaxer.output
    outputs=modded_softmax(T)(x)
    return tf.keras.Model(inputs,outputs)

def preprocessor():
    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_train,ds_test