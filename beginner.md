# Tensorflow 初體驗
## 安裝
用 anaconda 平台
* conda create --name tensorflow python=3.10.4
* 用 anaconda navigator 安裝
  * numpy 1.23.1
  * matplotlib 3.5.1
  * ipympl 0.8.7
  * opencv 4.5.5
  * tensorflow 2.8.2
* 但是如此的安裝，執行 beginner.ipynb 時，matplotlib 與 tensorflow 似乎有衝突，python  kernel 會當掉，參考 [tensorflow 網站](https://www.tensorflow.org/install)，改用 pip 安裝 tensorflow 2.9.1，在 tensorflow 環境(env)下作如下修改
  * 用 anaconda navigator 移除 tensorflow 2.8.2
  * pip install --upgrade pip
  * pip install tensorflow
## 安裝 tensorflowjs
[Github tfjs](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#regular-conversion-script-tensorflowjsconverter) 提到
```
The script pulls its own subset of TensorFlow, which might conflict with the existing TensorFlow/Keras installation.

Note: Check that tf-nightly-2.0-preview is available for your platform.

Most of the times, this means that you have to use Python 3.6.8 in your local environment.
```
因此，在python3.6.8環境下，順利安裝tensorflowjs，方法為 `pip install tensorflowjs`，它包含了tensorflow。接著用Anaconda Navigator 安裝了 numpy, matplotlib, ipympl, opencv。


## 將 python 訓練好的模型移植到瀏覽器使用
參考[How to Convert a Keras SavedModel into a Browser-based Web App](https://www.freecodecamp.org/news/convert-keras-savedmodel-into-browser-based-webapp/)，與[Importing a Keras model into TensorFlow.js](https://www.tensorflow.org/js/tutorials/conversion/import_keras)
* 假設new_prob_model 是python訓練好的模型，下面指令將new_prob_model 轉換成tensorflow.js可處理的格式存到Saved_Model/converted_tfjs_model目錄裡
  * import tensorflowjs as tfjs
  * tfjs.converters.save_keras_model(new_prob_model, "Saved_Model/converted_tfjs_model")
* 在網頁中，用下面指令進行辨識
  * model = await tf.loadLayersModel('saved_model/converted_tfjs_model/model.json');
  * model.predict()
  * 詳情見draw.html
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

## 製造開發階段用的 SSL 憑證
If you don't care about your certificate being trusted (usually the case for development purposes) you can just create a self-signed certificate. To do this, we can use almost the same line, but we'll pass two extra parameters.

`openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem`

注意，要使用webcam.js，必須是https網站。