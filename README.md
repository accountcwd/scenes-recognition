
### 数据集
>场景类别训练使用[place365数据集](http://places2.csail.mit.edu/demo.html) <br>
场景属性使用[SUN Attribute Database](http://cs.brown.edu/~gmpatter/sunattributes.html)<br>
[官方输出场景类别与属性推断示例](http://places2.csail.mit.edu/demo.html)<br>
### 模型
MobileNetV2 <br>
>[预训练模型](https://github.com/shicai/MobileNet-Caffe)<br>
推断时间:233ms<br>
模型大小：11.5M<br>
### 输出
>有两类输出：场景**类别**(365类)与场景**属性**(717种) <br>
输出有两个全连接层，分别是场景类别**置信度**与场景属性**分数**<br>
其中场景属性为多标签输出，分数阀值为1(我使用0.95)，即分值大于阀值就可以认为输入具有该属性。<br>

### 测试集准确率：
训练了60万ITER = 12epoch，测试准确率：<br>
accuracy = 0.523699<br>
accuracy_5 = 0.827315<br>
loss = 1.79646 (* 1 = 1.79646 loss)<br>
对比RESnet152 benchmark：the top1 accuracy is 54.74%(-2.37%) and the top5 accuracy is 84.98%(-2.25%).<br>
训练完*类别*后再冻结卷积层训练*属性*<br><br>
由于属性为多标签输出，而给的标注也较为主观，没法计算准确率，训练Loss值约为**8**，验证Loss值约为**9.1**。<br>






 




