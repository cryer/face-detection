# face-detection
      人脸领域有几个研究方向，分别是人脸检测，人脸校验，人脸验证和人脸识别。
      人脸检测主要是从图像中检测出人脸，用矩形框选中人脸。
      人脸校验主要是标记处人脸的特征点，比如眼睛，嘴角，鼻子，脸的轮廓等等。
      人脸验证就是比较两张脸是不是一个人。
      人脸识别本质上就是多次人脸验证，将人脸与库里的脸一一比对，找出最相近最匹配的脸。
    而我这次主要讨论的是人脸检测，这其实也是人脸识别的一部分，因为只有检测出人脸，接下来才能进行识别。
## 人脸检测
    首先讨论传统机器学习的人脸检测方法，参考zouxy09的博客。
### HOG特征
     方向梯度直方图（Histogram of Oriented Gradient, HOG）特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。
     它通过计算和统计图像局部区域的梯度方向直方图来构成特征。
     Hog特征结合SVM分类器已经被广泛应用于图像识别中，尤其在行人检测中获得了极大的成功。
#### 主要思想
    在一副图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。
    （本质：梯度的统计信息，而梯度主要存在于边缘的地方）。
#### 实现方法
    先将图像分成小的连通区域，我们把它叫细胞单元。然后采集细胞单元中各像素点的梯度的或边缘的方向直方图。
    最后把这些直方图组合起来就可以构成特征描述器。
#### 提高性能
    把这些局部直方图在图像的更大的范围内（我们把它叫区间或block）进行对比度归一化（contrast-normalized），
    所采用的方法是：先计算各直方图在这个区间（block）中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。
    通过这个归一化后，能对光照变化和阴影获得更好的效果。
#### 优点
    与其他的特征描述方法相比，HOG有很多优点。首先，由于HOG是在图像的局部方格单元上操作，
    所以它对图像几何的和光学的形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。
    其次，在粗的空域抽样、精细的方向抽样以及较强的局部光学归一化等条件下，只要行人大体上能够保持直立的姿势，
    可以容许行人有一些细微的肢体动作，这些细微的动作可以被忽略而不影响检测效果。
    因此HOG特征是特别适合于做图像中的人体检测的。
#### 具体
    这些都是虚的，下面是干货：
    HOG特征提取方法就是将一个image（你要检测的目标或者扫描窗口）：
    1）灰度化（将图像看做一个x,y,z（灰度）的三维图像）；
    2）采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）；
        目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音的干扰；
    3）计算图像每个像素的梯度（包括大小和方向）；主要是为了捕获轮廓信息，同时进一步弱化光照的干扰。
    4）将图像划分成小cells（例如6*6像素/cell）；
    5）统计每个cell的梯度直方图（不同梯度的个数），即可形成每个cell的descriptor；
    6）将每几个cell组成一个block（例如3*3个cell/block），一个block内所有cell的特征descriptor串联起来便得到该block的HOG特征descriptor。
    7）将图像image内的所有block的HOG特征descriptor串联起来就可以得到该image（你要检测的目标）的HOG特征descriptor了。
       这个就是最终的可供分类使用的特征向量了。
可以借助下面这张图来理解：

![](https://github.com/cryer/face-detection/raw/master/image/1.png)


### LBP特征
     LBP（Local Binary Pattern，局部二值模式）是一种用来描述图像局部纹理特征的算子；
     它具有旋转不变性和灰度不变性等显著的优点。它是首先由T. Ojala, M.Pietikäinen, 和D. Harwood 在1994年提出，
     用于纹理特征提取。而且，提取的特征是图像的局部的纹理特征；
#### 特征的描述
    原始的LBP算子定义为在3*3的窗口内，以窗口中心像素为阈值，将相邻的8个像素的灰度值与其进行比较，
    若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3*3邻域内的8个点经比较可产生8位二进制数
    （通常转换为十进制数即LBP码，共256种），即得到该窗口中心像素点的LBP值，并用这个值来反映该区域的纹理信息。
    如下图所示：
![](https://github.com/cryer/face-detection/raw/master/image/2.png)
#### 改进
    LBP算法相对简单了很多，而且出现了很多的改进，现在的LBP只是初始版本，存在很多不足，比如只满足灰度不变性，不满足旋转不变性，
    所以后来又出现了圆形LBP，以及其旋转不变模式等等。
### Haar特征
     Haar特征分为三类：边缘特征、线性特征、中心特征和对角线特征，组合成特征模板。
     特征模板内有白色和黑色两种矩形，并定义该模板的特征值为白色矩形像素和减去黑色矩形像素和。
     Haar特征值反映了图像的灰度变化情况。例如：脸部的一些特征能由矩形特征简单的描述，
     如：眼睛要比脸颊颜色要深，鼻梁两侧比鼻梁颜色要深，嘴巴比周围颜色要深等。
     但矩形特征只对一些简单的图形结构，如边缘、线段较敏感，所以只能描述特定走向（水平、垂直、对角）的结构。
 ### Haar-like特征介绍
 ![](https://github.com/cryer/face-detection/raw/master/image/3.png)
 
       对于图中的A, B和D这类特征，特征数值计算公式为：v=Sum白-Sum黑，而对于C来说，计算公式如下：v=Sum白-2*Sum黑；
       之所以将黑色区域像素和乘以2，是为了使两种矩形区域中像素数目一致。
       通过改变特征模板的大小和位置，可在图像子窗口中穷举出大量的特征。上图的特征模板称为“特征原型”；
       特征原型在图像子窗口中扩展（平移伸缩）得到的特征称为“矩形特征”；矩形特征的值称为“特征值”。
       矩形特征可位于图像任意位置，大小也可以任意改变，所以矩形特征值是矩形模版类别、矩形位置和矩形大小这三个因素的函数。
       故类别、大小和位置的变化，使得很小的检测窗口含有非常多的矩形特征，如：在24*24像素大小的检测窗口内矩形特征数量可以达到16万个。
       这样就有两个问题需要解决了：（1）如何快速计算那么多的特征？—积分图大显神通；
       （2）哪些矩形特征才是对分类器分类最有效的？—如通过AdaBoost算法来训练
