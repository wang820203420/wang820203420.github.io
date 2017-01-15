
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

<br>那我们来看看部分代码。demo 我会上传到 github 需要看源码的的可以[下载]。
<br>

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


<br>具体可以去看 [MNIST](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html) 这里面有详细的介绍，可以让你初步了解关于 tf 的运作过程。









 