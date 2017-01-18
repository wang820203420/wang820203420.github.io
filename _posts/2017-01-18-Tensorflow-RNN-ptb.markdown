
一切<br>始于无

<p id = "build"></p>
---



关于 tf 的 rnn ptb demo，遇到了很多坑，运行出错。后来听了老司机的建议，才发现 tf 的版本 和tf 的源码对不上号。我的是0.9的环境，那么也要对应 0.9的源码，果然是经验之谈，看到的同学要注意了，前面的坑我都踩了。

首先我们得[看看](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/recurrent.html)这是简单的介绍。
我们需要准备好 PTB 数据 [下载](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) 

<br>然后去[github tensorflow](https://github.com/tensorflow/tensorflow/releases?af) 找到 对应自己环境的 源码，下载下来。

<br>我是用的 pycharm 这个idea，你们也可以下载个。

<br>然后苦于每次 pip 安装tf的时候慢的狠，我把 pip 源改成了淘宝的，居然发现，秒下啊。安装好tf 的环境，把源码放在里面。

<br>我们还记得首先下载好的 ptb 数据集么？我们现在解压，然后在cd 到源码的 ptb 文件目录下输入：python ptb_word_lm.py --data_path=simple-examples/data/ --model small 就行了。

<br>model small 是调用代码里的小模型，还有中等、大的，根据自己电脑环境来。simple-examples 就是下载好的 ptb 数据集。

<br>
> Hello

![img](/img/in-post/ptb.png)


---


我的还在运行中，估计还要好久，耐心等待下。这几天准备把 tf 的 demo 都实践跑一遍，实践出真理啊，遇到不懂的或者出错去查，学起来事半功倍。






 