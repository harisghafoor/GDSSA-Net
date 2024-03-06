# Official Implementation of Gradually Deeply Supervised Self-Ensemble Attention Based Network for Thyroid Nodule Segmentation 

Implementation of paper - [Title of Paper](https://arxiv.org/abs/2207.02696)

<!-- <a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> -->
<!-- [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696) -->

<div align="center">
    <a href="./">
        <img src="./figures/GDSSA-Net Diagram.png" width="79%"/>
    </a>
</div>

## Performance 
### Training Performance: 
<div align="center">
    <a href="./">
        <img src="./figures/training_curves.png" width="79%"/>
    </a>
</div>

### Testing Performance : 
<div align="center">
    <a href="./">
        <img src="./figures/ROC.png" width="79%"/>
    </a>
</div>

## Installation

Conda environment (recommended)
<!-- <details><summary> <b>Expand</b> </summary> -->

``` shell
conda env create -f environment.yml
```

</details>

## Testing

[`model.pth`](https://drive.google.com/drive/folders/1BsYMs5CzG0CKHDCOHrYF7HKHpfA5newQ?usp=sharing)

``` shell
python src/test.py --fold fold_2 --experiment_name Thyroid_Segmentation_Experiment --checkpoint_path checkpoints/model.pth --device cpu --thresholds 0.5 0.1 0.1 0.5 --json_file tn3k_combo_folds.json
```

## Training

Data preparation

``` shell
bash scripts/get_data.sh
```

* Download the data from the drive ([Link](https://drive.google.com/drive/folders/1UIaCimog2VWt3iHEGDCG7o2ZCsSAMi7E?usp=drive_link)

Training

``` shell

python src/train.py --data_dir datasets/Thyroid\ Dataset/tn3k --output_dir output --experiment_name Thyroid_Segmentation_Experiment --batch_size 16 --num_epochs 100 --device cpu --PARENT_DIR BestFoldAttentionUnetDDTI --augmented_data augmented_data_ddti --fold fold_2

```


## Citation
<details><summary> <b>Expand</b> </summary>

<!-- ```
@inproceedings{harisgdssa2024,
  title={{GDSSA}: Gradually Deeply Supervised Self-Ensemble Attention Based Network for Thyroid Nodule Segmentation},
  author={haris,usman,umar,azka and -},
  journal={Bio-engineering,2024 (MDPI)},
  year={2024}
}

``` -->
</details> -->



## Example Visualizations

TN3k Dataset : 

<div align="center">
    <a href="./">
        <img src="./figures/results_ddti.png" width="80%"/>
    </a>
</div>


<!-- ## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details> -->
