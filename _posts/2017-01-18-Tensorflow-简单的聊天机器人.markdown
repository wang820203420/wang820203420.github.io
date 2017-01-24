
拨云<br>重复开

<p id = "build"></p>
---

一直想做一个聊天机器人 ai。于是乎刚好看见了[熊猫大哥的博客](http://blog.topspeedsnail.com/archives/10735)，动手跟着实现了下，很有意思。

<br>也是先准备好[训练数据集](https://github.com/rustch3n/dgk_lost_conv) 用sublime把dgk_shooter_min.conv转成utf8 格式。然后数据预处理下把对话分成问答。代码我这里就不贴了。

<br>他用的也是 seq2seq_model.py 模型，这个模型原本是tf机器翻译demo 里那个模型，我大概训练了差不多1天1夜，到了11000步，用的 cpu。玩这个最起码你得要有耐心，耐心的等待一下。不过这个demo无论你怎么训练都结果都不是很好，单纯的使用数据而已。你可以训练一会然后终止训练测试下。 我的是py2.7的环境，在终端中要“你好”这样的语法来写，不然会报错。py 3 的环境可以去掉 “”。

<br>最后得到的测试结果，很蠢的机器人。






<br>
> Hello

![img](/img/in-post/robot.PNG)


---









 