import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape= [28, 28, 1], kernel_size= 5, filters= 8, strides= 1, 
                   activation= 'relu',  kernel_initializer= 'variance_scaling'),
        tf.keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2]),
        tf.keras.layers.Conv2D(kernel_size= 5, filters= 16, strides= 1, activation= 'relu', kernel_initializer= 'variance_scaling'),
        tf.keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2]),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(units= 128, kernel_initializer= 'variance_scaling', activation= 'relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units= 10, kernel_initializer= 'variance_scaling', activation= 'softmax')
    ])
    model.compile(optimizer = 'adam', 
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics = ['accuracy'])
    return model

def training():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train, x_train.shape+(1,))
    x_test=np.reshape(x_test, x_test.shape+(1,))
    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = create_model()
    epochs = 30
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    model.evaluate(x_test,  y_test, verbose=2)
    import matplotlib.pyplot as plt
    x = [a for a in range(epochs)]
    fig, ax = plt.subplots()
    ax.plot(x, history.history['accuracy'], label="acc")
    ax.plot(x, history.history['val_accuracy'], label="val_acc")
    ax.legend()
    ax.set_xlabel("epoch")
    plt.show()

    tfjs.converters.save_keras_model(model, "digit_cnn_model")

if __name__ == "__main__":
    training()
