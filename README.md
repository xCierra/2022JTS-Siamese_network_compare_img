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
    ── annos.csv
```
### 读取数据 LoadData
```python
def loadData(datadir,type):
    img1_path=[]
    img2_path=[]
    labels=[]
    # 获取train/test路径下的文件夹
    for i in os.listdir(datadir):
        # 路径拼接获取train路径下文件夹里的文件
        for fn in os.listdir(os.path.join(datadir, str(i))):
            # 判断后缀是否为jpg
            if fn.endswith('.jpg'):
                # 拼接完整的文件路径
                fd = os.path.join(datadir, str(i), fn)
                # print(fd)
                # 切分文件夹两张图片的路径分别放到img1_path、img2_path
                if os.path.split(fd)[1]=='a.jpg':
                    img1_path.append(fd)
                else:
                    img2_path.append(fd)

                # 按照8:2的比例，前480为训练集，后120为测试集，在导入数据前，需要手动把数据放到对应的文件夹，参考数据结构
                annos=pd.read_csv('init_data/data/annos.csv')
                
                # 如果类型是'train'读取csv文件标签列前480行
                # 如果类型是'test'读取csv文件标签列后120行
                if type == 'train':
                    labels=annos['label'][:480]
                    labels.append(labels)
                else:
                    labels=annos['label'][480:]
                    labels.append(labels)

    return img1_path, img2_path,np.array(labels)

# 获取img1和img2的路径和label
x1_train,x2_train,y_train=loadData(train_dir,'train')
x1_test,x2_test,y_test,=loadData(test_dir,'test')
```
## 模型训练
### 导入图片并转化为张量 load_img to tensor
