# [CVPR 2024] Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis
## 1. Introduction
### WiKG: *Whole Slide Image is a **K**nowledge Graph*

![WiKG Framework](figs/wikg_main.png "The framework of our proposed method for WSI analysis, including patch feature extraction, dynamic edge construction based on head and tail embeddings, graph representation learning, and the prediction of WSIs.")

We demonstrate a novel whole slide image (WSI) analysis method based on graph representation called *WiKG*, which represents a WSI as a knowledge graph, cropped patches as graph nodes, and uses the head-to-tail embedding of patches to generate dynamic graph representations. 

## 2. Reproduce WiKG
This repository is based on the Pytorch version of the WiKG code. 

The easy-to-follow model code and train demo code have been released, and the details are as follows:

### 2.1 Data Structure
The Datasets we used can be downloaded from [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga). How to crop WSIs into patches can refer to [sdpc-for-python](https://github.com/WonderLandxD/sdpc-for-python). And how to extract the initial features of WSIs can refer to [CLAM](https://github.com/mahmoodlab/CLAM). Assume that we have divided the dataset into 5 folds and stored them in the following structure:
```bash
DATASET/
	├── fold0
    		├── train_data.csv
            ├── val_data.csv
	├── fold1
    		├── train_data.csv
            ├── val_data.csv
	├── fold2
    		├── train_data.csv
            ├── val_data.csv
	├── fold3
    		├── train_data.csv
            ├── val_data.csv
	├── fold4
    		├── train_data.csv
            ├── val_data.csv
```
Where each .csv file stores the storage path of the initial features of each WSI.

### 2.2 Train and Test

``` shell
python train.py 
```
The above command will train and test WiKG.

### 2.3 TCGA Dataset used in paper
The detailed data ID and label division in tcga used in the article have been updated to the data folder.

## 3. About Paper

Arxiv version: https://arxiv.org/abs/2403.07719

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://arxiv.org/abs/2403.07719)
```
@inproceedings{li2024dynamic,
  title={Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis},
  author={Li, Jiawen and Chen, Yuxuan and Chu, Hongbo and Sun, Qiehe and Guan, Tian and Han, Anjia and He, Yonghong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11323--11332},
  year={2024}
}
```




-----------------------
*Jiawen Li, H&G Pathology AI Research Team*

