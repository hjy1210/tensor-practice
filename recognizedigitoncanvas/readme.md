# 辨識瀏覽器畫布上的數字
大家都知道深度學習(Deep learning)可以用來辨識手寫數字。

本文打算實做網頁，使用者在上面的畫布(canvas)寫數字，由程式辨識出數字來。

本文試圖利用深度學習的平台 TensorFlow 來進行。

## 在畫布上畫數字
利用畫布上的mousedown, mousemove, mouseup 事件來取得筆畫的軌跡，利用畫布之 context 的 moveTo, lineTo來畫畫。

為了要能在手機上執行，將touchstart、touchmove、touchend事件裡分別觸發人造的mousedown, mousemove, mouseup事件，
並用preventDefault防止在畫布上寫字時造成整個螢幕的移動。

## 畫布上的圖像轉成TensorFlow.js所需要的Tensor
用 tf.browser.fromPixels(canvas,3) 指令將畫布canvas上的圖像轉成[w,h(=w),3]的三階Tensor，假設圖像都是灰階，
只取紅色即為所要的灰階，再轉成[1,w,w,1]形狀，再用resizeBiLinear調整大小為[1,28,28,1]的形狀，
這是TensorFlow模型所要的輸入素材。

## javascript 輸入在Python訓練好的模型
用tf.loadLayersModel('saved_model/digit_cnn_model/model.json')會得到模型的Promise，
其中model.json與一起的的參數資料是用Python訓練好的模型再轉成Tensorflow.js所要的格式。

## Python 輸出合乎 javascript 格式的模型
用指令 tfjs.converters.save_keras_model(model, "saved_model/digit_cnn_model") 將訓練好的模型model存成Tensorflow.js所要的格式放在saved_model/digit_cnn_model目錄中，裡面有model.json與group1-shard1of1.bin。

## 在Python裡用Tensorflow訓練模型


## DigitRecognizer.html

## 瀏覽器上的畫布
首先，需要了解如何存取畫布上的像素。

[Pixel_manipulation_with_canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Pixel_manipulation_with_canvas)裡告訴我們存取的方法。假設canvas 是網頁裡的畫布，canvas.imageData.data 就是一維的像素陣列Uint8ClampedArray，內含 width x height x 4，其中width,height分別是畫布的寬與高，4代表每一像素有紅(R)、綠(G)、藍(B)與透明度(A)四個值。

## draw.html
[E-Signature using canvas](https://codepen.io/yguo/pen/OyYGxQ)

[bencentra/esignature.html](https://gist.github.com/bencentra/91350fe91c377c1ca574)
