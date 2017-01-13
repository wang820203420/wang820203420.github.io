
坐立不安<br>青蛙抓不到虫子

<p id = "build"></p>
---

<br>
> Hello

![img](/img/in-post/shibie1.png)


---

跑了tf 的minist 训练感觉蛮有意思的，然后试了下 TensorFlow 里面有个项目叫 **camera**，可以进行物体识别。玩这个项目首先你得 [下载tf 项目]( https://github.com/tensorflow/tensorflow) 在 contrib 文件下找到 ios_examples 就可以看到了。

<br>当然现在是运行不起来的，你还需要3个文件。
<li> libtensorflow-core.a
<li> imagenet_comp_graph_label_strings.txt
<li> tensorflow_inception_graph.pb

<br>libtensorflow-core.a 这个需要自己编译<br>imagenet_comp_graph_label_strings.txt 这个是文本数据，用来识别物体时显示出对应的文字。

<br>tensorflow_inception_graph.pb 这个是 Google 训练好的模型。

<br>有了这三个添加到工程中你就可以运行了。

<br>那我们来看看部分代码。demo 我会上传到 github 需要看源码的的可以下载。
## demo
> -(void)viewDidLoad {
  <br>[super viewDidLoad];
  <br>square = [[UIImage imageNamed:@"squarePNG"] retain];
  <br>synth = [[AVSpeechSynthesizer alloc] init];//文本语音转换
  <br>labelLayers = [[NSMutableArray alloc] init];
  <br>oldPredictionValues = [[NSMutableDictionary alloc] init];
  //载入文本和模型文件
  <br>tensorflow::Status load_status;//载入tf
  <br>if (model_uses_memory_mapping) {//如果没有模型
  <br>load_status = LoadMemoryMappedModel(//载入内存映射模型
        <br>model_file_name, model_file_type, &tf_session, &tf_memmapped_env
<br>);
  <br>}else {
    <br>load_status = LoadModel(model_file_name, model_file_type, &tf_session);
  <br>}
  <br>if (!load_status.ok()) {
    LOG(FATAL) << "Couldn't load model: " << load_status;
  }

 >tensorflow::Status labels_status =
      LoadLabels(labels_file_name, labels_file_type, &labels);
 <br>if (!labels_status.ok()) {
    <br>LOG(FATAL) << "Couldn't load labels: " << labels_status;
  <br>}
    //启动图像采集
  <br>[self setupAVCapture];
<br>}
NHJ010901


<br>其实它的整个过程，从采集到图像，载入到训练好的模型中。利用多重卷积神经网络提取出特征点，然后逐步分析出是杯子的概率多少，是狗的概率多少。如果杯子的概率高，文本直接转成语音。

<br>当然里面卷积神经是怎么操作的，我们来看看 tf mnist 数据集,我会逐步分析出它们是怎么运作的。

<br>首先我们要加入训练好的 mnist 数据集，这个数据集里面都是谷歌的训练手写数字，每张图片都是28*28像素点。我们在处理的过程中，会将图片展开成 28 * 28 = 784 的向量。用这个方式来表达这张图片。

<br>创建计算图，在session 中启动它，它随后会用 NumPy 科学计算库来完成计算。

<br>然后我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。定义好权重和偏置量。利用 softmax 函数计算概率。还得用 reduce_sum 函数指定最小误差的损失函数，将纬度求和。关于这些函数，我也没有太深入，内部是怎么运作的，抽空研究下。

<br>然后就可以开始训练模型来，设置步长，用最速下降法。防止样本重复过拟合，增加了迭代的效率。

<br>每次加入50 个样本，迭代10000次，然后评估模型，输出打印结果。

当然我用的 cpu 跑的，有条件的可以用 gpu跑。哈哈。

<br>91%的准确率。后面我又增加了多层神经网络，提高到了93%，但是没达到官网 97%。


<br>
# load MNIST data
import input_data
from mnist_demo import *
mnist = input_data.read_data_sets("mnist", one_hot=True)



# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()



># weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

>def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

># convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
># pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

># Create the model
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

>y = tf.nn.softmax(tf.matmul(x,W) + b)

># first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

>x_image = tf.reshape(x, [-1, 28, 28, 1])

>h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
>h_pool1 = max_pool_2x2(h_conv1)

># second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

>h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
>h_pool2 = max_pool_2x2(h_conv2)

># densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

>h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

># dropout
>keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

># readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

>y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

># train and evaluate the model
<br>cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
<br>train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

><br>for i in range(100):
    <br>batch = mnist.train.next_batch(50)
    <br>if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        <br>print "step %d, train accuracy %g" %(i, train_accuracy)
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})



>print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})







 