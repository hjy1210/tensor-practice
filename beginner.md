# Tensorflow 初體驗
## 安裝
用 anaconda 平台
* conda create --name tensorflow python=3.10.4
* 用 anaconda navigator 安裝
  * numpy 1.23.1
  * matplotlib 3.5.1
  * ipympl 0.8.7
  * tensorflow 2.8.2
* 但是如此的安裝，執行 beginner.ipynb 時，matplotlib 與 tensorflow 似乎有衝突，python  kernel 會當掉，參考 [tensorflow 網站](https://www.tensorflow.org/install)，改用 pip 安裝 tensorflow 2.9.1，在 tensorflow 環境(env)下作如下修改
  * 用 anaconda navigator 移除 tensorflow 2.8.2
  * pip install --upgrade pip
  * pip install tensorflow

## 練習 beginner.py
```
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
import numpy as np
p= probability_model(x_test[:10])
print(p)
v = np.argmax(p,axis=-1)
from matplotlib import pyplot as plt
for i in range(10):
    plt.imshow((x_test[i]*255).astype('uint8'), cmap=plt.cm.binary)
    print(y_test[i], v[i])
    plt.show()
    #plt.pause(0.1)
```
