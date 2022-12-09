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
```python
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
```
#### 对比损失
```python
def contrastive_loss(y_true, y_pred):
    margin = 1.
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
```
#### 采用resnet网络进行训练，加速收敛，权重是imagenet
```python
def base_net(input_tensor_shape):
    input = Input(input_tensor_shape)
    conv_base = ResNet152V2(weights='imagenet',
                         include_top=False)
    conv_base.trainable = False
    net = conv_base(input)
    net = layers.Flatten()(net)
    net=layers.Dropout(0.1)(net)
    net = layers.Dense(512, activation='relu')(net)
    return keras.Model(input, net)
```
#### 计算准确度
```python
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
```
## 数据准备
### 数据结构
```python
── init_data
    ── data
      └── train
        ├── folds
          └── a.jpg
          └── b.jpg
      └── test
         ├── folds
           └── a.jpg
           └── b.jpg
    annos.csv
```
