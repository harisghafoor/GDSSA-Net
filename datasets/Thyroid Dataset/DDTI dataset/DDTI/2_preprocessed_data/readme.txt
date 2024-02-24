1) The data in folder "stage1" is used to train the first network, which contains the preprocessed image with no irrelevant regions;
2) And "stage2" is used to train the second network,  which contains the image in the expanded ROI.

Please visit the following link for more details：https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
--------------------------------------------------------------------------------------------------------------------
1）stage 1这个文件夹是用来训练级联框架中第一个粗分割网络的，是经过预处理去除irrelevant regions的图片，所有图片都被resize到同一尺寸，一般为256；
2）stage 2这个文件夹是用来训练级联框架中第二个细分割网络的，是基于结节分割金标准进行一定外扩的ROI内的图像。

更多细节详见：https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
