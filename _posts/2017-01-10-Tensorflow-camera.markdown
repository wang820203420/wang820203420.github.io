
坐立不安<br>青蛙抓不到虫子

<p id = "build"></p>
---



跑了tf 的minist 训练感觉蛮有意思的，然后试了下 TensorFlow 里面有个项目叫 **camera**，可以进行物体识别。玩这个项目首先你得 [下载tf 项目]( https://github.com/tensorflow/tensorflow) 在 contrib 文件下找到 ios_examples 就可以看到了。

<br>当然现在是运行不起来的，你还需要3个文件。
<li> libtensorflow-core.a
<li> imagenet_comp_graph_label_strings.txt
<li> tensorflow_inception_graph.pb
<br>libtensorflow-core.a 这个需要自己编译，网上有教程可以自己去查查这里就不说了。
<br>imagenet_comp_graph_label_strings.txt 这个是文本数据，用来识别物体时显示出对应的文字。

<br>tensorflow_inception_graph.pb 这个是 Google 训练好的模型。

<br>有了这三个添加到工程中你就可以运行了。


<br>
> Hello
这里是效果。

![img](/img/in-post/IMG_9713.png)
![img](/img/in-post/IMG_9714.png)


---





 