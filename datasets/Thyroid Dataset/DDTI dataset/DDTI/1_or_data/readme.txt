This folder contains the secondary processed DDTI public thyroid data. We have processed the DDTI dataset into the format of 2020 TN-SCUI challenge dataset. The "image" folder contains ultrasound images, and the "mask" folder contains the gold standard for segmentation of thyroid nodules.

 - TNSCUI official website: https://tn-scui2020.grand-challenge.org/Home/
 - DDTI data set document: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/1/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.short
 - DDTI data set download link: http://cimalab.intec.co/applications/thyroid/

!!!!!Caution:
1) This data has been processed twice by us, which is different from the original data. Specifically, we have removed some images marked with damage and images with multiple nodules in a single scan;
2) We have renumbered all images, that is, the image number is not the image number in the original database;
2) The category information in the "category.csv" file is forged and cannot be used for classification tasks, so this data set is only for segmentation tasks.

Sincere thanks to the open access DDTI database
------------------------------------------------------------------------------------------------------------------------------------------

本文件夹为DDTI公开甲状腺数据，其中image文件夹包含了超声图像，mask文件夹包含了甲状腺结节分割金标准。

 - TN-SCUI官网：https://tn-scui2020.grand-challenge.org/Home/
 - DDTI数据集文档：https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/1/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.short
 - DDTI数据集下载链接：http://cimalab.intec.co/applications/thyroid/ 

!!!!!需要注意的是：
1）这个数据是经过我们二次处理的数据，与原始数据有一定的不同，具体来说，我们去除了一些标注有损坏的图像，以及单张扫描有多个结节的图像；
2）我们对所有图像进行了重新编号，即图像序号并不是原始数据库中的图像编号；
2）category.csv这个文件中类别信息是伪造的，不可以用作分类任务，因此此数据集仅供分割任务使用。

由衷的感谢DDTI开源数据库