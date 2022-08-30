# 辨識瀏覽器畫布上的數字
大家都知道深度學習(Deep learning)可以用來辨識手寫數字。

本文打算實做網頁，使用者在上面的畫布(canvas)寫數字，由系統辨識出數字來。

本文試圖利用深度學習的平台 TensorFlow 來進行。

## 瀏覽器上的畫布
首先，需要了解如何存取畫布上的像素。

[Pixel_manipulation_with_canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Pixel_manipulation_with_canvas)裡告訴我們存取的方法。假設canvas 是網頁裡的畫布，canvas.imageData.data 就是一維的像素陣列Uint8ClampedArray，內含 width x height x 4，其中width,height分別是畫布的寬與高，4代表每一像素有紅(R)、綠(G)、藍(B)與透明度(A)四個值。
