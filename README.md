## Siamese:孪生神经网络比较图像相似性
---
## 目录
1. [所需环境 Environment](#所需环境)
2. [定义网络 Network](#定义网络)
3. [数据准备 Data Preparation](#数据准备)
4. [模型训练 Train](#模型训练)
5. [模型推理 Predict](#模型推理)
6. [注意事项 Attention](#注意事项)

## 所需环境
keras==2.10.0
numpy==1.21.3
opencv_python==4.5.4.58
pandas==1.3.4
tensorflow_gpu==2.10.0

## 定义网络
### 构建Siamese网络
#### 欧式距离
`def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))`
