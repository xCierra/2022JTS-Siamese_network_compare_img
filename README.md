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
#### 数据结构 
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
#### 读取数据
```python
#训练集
train_dir = 'init_data/data/train/'
#测试集
test_dir = 'init_data/data/test/'

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
#### 导入图片并转化为张量 load_img to tensor
```python
def get_tensor(image1_list, image2_list,label_list):
    img1 = []
    img2 = []
    for image1 in image1_list:
        #读取路径下的图片
        x = tf.io.read_file(image1)
        #将路径映射为照片,3通道
        x = tf.image.decode_jpeg(x, channels=3)
        #修改图像大小为(64,64)
        x = tf.image.resize(x,[64,64])
        #将图像压入列表中
        img1.append(x)
    for image2 in image2_list:
        #读取路径下的图片
        x = tf.io.read_file(image2)
        #将路径映射为照片,3通道
        x = tf.image.decode_jpeg(x, channels=3)
        #修改图像大小(64,64)
        x = tf.image.resize(x,[64,64])
        #将图像压入列表中
        img2.append(x)
    #将列表转换成tensor类型
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    y = tf.convert_to_tensor(label_list)
    return img1,img2,y
```
#### 对数据进行归一化
```python
def preprocess(x1,x2,y):
    x1 = tf.cast(x1,dtype=tf.float32) / 255.0
    x2 = tf.cast(x2,dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.float32)
    return x1,x2,y

# 把训练集转化为张量
x1_train,x2_train,y_train = get_tensor(x1_train,x2_train,y_train)
x1_train,x2_train,y_train = preprocess(x1_train,x2_train,y_train)

# 把测试集转化为张量
x1_test,x2_test,y_test = get_tensor(x1_test,x2_test,y_test)
x1_test,x2_test,y_test = preprocess(x1_test,x2_test,y_test)
```
#### 定义模型输入
```python
# 两张图片输入的尺寸为(64,64,3)
input_a=Input(shape=(64,64,3))
input_b=Input(shape=(64,64,3))

# 输入模型张量的尺寸为(64,64,3)
base_network=base_net((64,64,3))

# 把两张图片输入进模型
processed_a=base_network(input_a)
processed_b=base_network(input_b)

# 定义 lambda layer 实现欧式距离计算
merge_layer = layers.Lambda(euclidean_distance)([processed_a, processed_b],tf.float32)

# 定义批标准化层
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)

# 全连接激活函数
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
```
#### 创建模型
```python
siamese = keras.Model([input_a, input_b], outputs=output_layer)
```
#### 回调函数
```python
callbacks_list=[ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=10,verbose=1),
                ModelCheckpoint(filepath='code\\model\\model_weight.h5',monitor='val_loss',save_best_only=True)]
```
#### 模型编译
```python
siamese.compile(optimizer=Adam(lr=0.01),loss=contrastive_loss,metrics=[accuracy])
```
#### 开始训练
```python
history = siamese.fit(
      [x1_train,x2_train],y_train,
    steps_per_epoch=8,
    validation_data=([x1_test,x2_test],y_test),
    validation_steps=2,
    batch_size=64,
    callbacks=callbacks_list,
      epochs=100)
```
## 模型推理
#### 导入权重
```python
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'model\\model_weight.h5')
def base_net(input_tensor_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(input_tensor_shape)
    conv_base = ResNet152V2(weights='imagenet',
                         include_top=False)
    conv_base.trainable = False
    net = conv_base(input)
    net = layers.Flatten()(net)
    net=layers.Dropout(0.4)(net)
    net = layers.Dense(512, activation='relu')(net)
    return keras.Model(input, net)

input_a=Input(shape=(64,64,3))
input_b=Input(shape=(64,64,3))

base_network=base_net((64,64,3))
processed_a=base_network(input_a)
processed_b=base_network(input_b)
merge_layer = layers.Lambda(euclidean_distance)([processed_a, processed_b],tf.float32)
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)

# 创建模型
siamese = keras.Model([input_a, input_b], outputs=output_layer)
# 模型编译
siamese.compile(optimizer=Adam(lr=0.01),loss=contrastive_loss,metrics=[accuracy])
# 导入权重
siamese.load_weights(model_path)
```
#### 模型推理
```python
# 读取测试集
test_dir = 'init_data/data/test/'
x1_test,x2_test,y_test,= loadData(test_dir,'test')
d={'x1':x1_test,'x2':x2_test,'label':y_test}
df=pd.DataFrame(d)

# 把输入图片转化为张量
x1_test,x2_test,y_test = get_tensor(x1_test,x2_test,y_test)
x1_test,x2_test,y_test = preprocess(x1_test,x2_test,y_test)

# 模型推理
result = siamese.predict([x1_test,x2_test])
df['predict']=result
df.to_csv('result\\result.csv',index=None)
```
## 注意事项
### 文件路径问题
```python
# 因为数据和代码文件不在同一个目录下，如需要运行代码，需要在代码里加上以下代码切换路径
# 获取当前路径
os.getcwd()
# 切换上级路径
upDir = os.path.pardir
os.chdir(upDir)
```
