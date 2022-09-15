# tensorflow

## tensorflow with python
tensorflow 官網示範的第一個例子[TensorFlow 2 quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner)，以mnist為資料，只用flatten, dense, dropout等layer做了一個簡單的模型，用來判讀手寫阿拉伯字。`beginner.ipynb` 根據它做了練習增加對它的了解。

## tensorflow with javascript
tensorflow.js 官網示範了一個例子[Handwritten digit recognition with CNNs](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)，以 sprite file 的形式提供手寫數字的資料(避免網頁的瓶頸-頻繁的資料IO)，用了Convolution的技術。`handwrittendigits` 資料夾的內容以此為據開始撰寫。使用了conv2d,maxPooling2d,conv2d,maxPooling2d,flatten,dense等layers作了模型，並將訓練完的模型存成 `hdrcnn.json` 以及對應的 `bin`。

`Beginner-CNN.ipynb` 仿上面的模型在python裡訓練模型，最後參考[Use the Python API to export directly to TF.js Layers format](https://www.tensorflow.org/js/tutorials/conversion/import_keras)，用tfjs.converters.save_keras_model(new_prob_model, "Saved_Model/converted_tfjs_model")將訓練好的模型轉換成可讓tensorflow.js使用的形式放在 `converted_tfjs_model` 資料夾中。

更進一步，`digit_cnn.py` 在 flatten與dense之間加了dropout，將訓練好的模型轉換成可讓tensorflow.js使用的形式放在 `digit_cnn_model` 資料夾中。

## DeepLearning.AI TensorFlow Developer Professional Certificate
Coursera 上面的 DeepLearning.AI TensorFlow Developer Professional Certificate 由四門課程組成，分別是
* [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow?specialization=tensorflow-in-practice)
* [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow?specialization=tensorflow-in-practice)
* [Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow?specialization=tensorflow-in-practice)
* [Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction?specialization=tensorflow-in-practice)
  
這四門的 `python` 程式碼與作業在 [tensorflow-1-public](https://github.com/https-deeplearning-ai/tensorflow-1-public)，
### Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
`Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning` 這門課分四週。
* 第一週介紹了 tf.keras.Sequential, keras.layers.Dense, model.compile, model.fit, model.predict。
* 第二週介紹了fashion_mnist, tf.keras.layers.Flatten, tf.nn.relu, tf.nn.softmax, model.evaluate, tf.optimizers.Adam, sparse_categorical_crossentropy
* 第三週介紹了 tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D, 用來增加分析影像的精確度。分析 fashion_mnist 的正確度由88%提高到90%。其中`C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb` 利用 [layer.output for layer in model.layers] 來觀察 convolution 的效果很值得細細體會。`C1_W3_Lab_2_exploring_convolutions.ipynb` 說明了`Conv2D` 與 `MaxPooling2D` 功效。
* 第四週介紹了 `ImageDataGenerator`, `horse and human` 資料集, `keras.preprocessing.image.load_image`, `keras.preprocessing.image.img_to_array`。`C1_W4_Lab_2_image_generator_with_validation.ipynb` 示範，在 model.fit 時使用`validation_data`，可以了解 overfiting 是否發生。

## TensorFlow: Data and Deployment Specialization
Coursera 上面的 `TensorFlow: Data and Deployment Specialization` 由四門課程組成，分別是
 * [Browser-based Models with TensorFlow.js](https://www.coursera.org/learn/browser-based-models-tensorflow?specialization=tensorflow-data-and-deployment)
 * [Device-based Models with TensorFlow Lite](https://www.coursera.org/learn/device-based-models-tensorflow?specialization=tensorflow-data-and-deployment)
 * [Data Pipelines with TensorFlow Data Services](https://www.coursera.org/learn/data-pipelines-tensorflow?specialization=tensorflow-data-and-deployment)
 * [Advanced Deployment Scenarios with TensorFlow](https://www.coursera.org/learn/advanced-deployment-scenarios-tensorflow?specialization=tensorflow-data-and-deployment)

這四門的 js，html 檔案與作業在 [tensorflow-2-public](https://github.com/https-deeplearning-ai/tensorflow-2-public)

### Browser-based Models with TensorFlow.js
`Browser-based Models with TensorFlow.js` 這門課分四週。
* 第一週用線性回歸來介紹 tensorflow.js 的基本語法。語法與python類似，不過參數用object語法，又因要在瀏覽器上執行，所以經常需要使用`async/promise` 語法免得阻擋了UI執行緒。介紹 iris.csv, tf.data.csv, model.fitDataset, dataSync()。
* 第二週，以MnistData為資料，用conv2d, maxPoolinged, conv2d, maxPoolinged, flatten, dense, dense 為模型(比[tensorflow.js 官網的例子](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)多了一層dense)，訓練判別手寫數字。_**並讓讀者可以在畫布上寫字再由系統判讀**_。
* 第三週，介紹
  * 直接使用現成的模型toxicity
  * 直接使用現成的模型mobilenet
  * 從python訓練好的模型，轉換成可讓tensorflow.js使用的格式。
