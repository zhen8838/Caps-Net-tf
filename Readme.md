# 一个简单的Capsule Net

简单复现了一下`Capsule Net`,直接执行`train.py`即可.

    需要 tensorflow 1.13 tqdm
    
# 结果
![](res/1.png)
![](res/2.png)

# 问题

我将`CapsDense`写成了`keras.Layers`,但是目前无法接受不定长度的输入,所以无法在`tf 2.0`中很好用起来,所以我又退回到了`tf 1.13`实现.这个有待以后解决.