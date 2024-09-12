import tensorflow as tf

def load_model():
    # Code to load your pre-trained MNIST model
    model = tf.keras.models.load_model("C:/Users/HP/Desktop/MNIST/my_cnn_model.h5")
    return model