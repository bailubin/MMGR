# MMGR
Official implementation of the paper "Geographic mapping with unsupervised multi-modal representation learning from VHR images and POIs", a multimodal self-supervised contrastive learning framwork using very high resolution (VHR) images and POIs. 
> **Abstract.**
> Most supervised geographic mapping methods with very-high-resolution (VHR) images are designed for a specific task, leading to high label-dependency and inadequate task-generality. Additionally, the lack of socio-economic information in VHR images limits their applicability to social/human-related geographic studies. To resolve these two issues, we propose an unsupervised multi-modal geographic representation learning framework (MMGR) using both VHR images and points-of-interest (POIs), to learn representations (regional vector embeddings) carrying both the physical and socio-economic properties of the geographies. In MMGR, we employ an intra-modal and an inter-modal contrastive learning module, in which the former deeply mines visual features by contrasting different VHR image augmentations, while the latter fuses physical and socio-economic features by contrasting VHR image and POI features. Extensive experiments are performed in two study areas (Shanghai and Wuhan in China) and three relevant while distinctive geographic mapping tasks (i.e., mapping urban functional distributions, population density, and gross domestic product), to verify the superiority of MMGR. The results demonstrate that the proposed MMGR considerably outperforms seven competitive baselines in all three tasks, which indicates its effectiveness in fusing VHR images and POIs for multiple geographic mapping tasks. Furthermore, MMGR is a competent pre-training method to help image encoders understand multi-modal geographic information, and it can be further strengthened by fine-tuning even with a few labeled samples.
> ![流程图10](https://github.com/bailubin/MMGR/assets/29422469/1b643f36-1fca-48ff-b229-5c05d105b0d1)
# requirements
```bash
torch
torchvision
torch_geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
```
# Usage
## Dataset 
Due to the copyright restrictions, we sample 200 samples (including image patches and the corresponding POI embeddings) from Shanghai and release them for model training simulation.  
Baiduyun drive:  
address：[https://pan.baidu.com/s/1xza04ceKHNfai77yUY4Zwg](https://pan.baidu.com/s/1xza04ceKHNfai77yUY4Zwg) 
code：c6zp  
Google drive:  
https://drive.google.com/drive/folders/1gU3yPyz0l59zrJNXCHTNbSwPXuYnVwVi?usp=drive_link  

## POI category representation learning
Users need to first get the POI category embeddings, and we utilize the [SemanticPOIEmbedding](https://github.com/RightBank/Semantics-preserved-POI-embedding) to embed the POIs.
## train and evaluate MMGR
```bash
# train mmgr model
python mmgr.py --data_path the_path_to_your_img_and_poi_data --total_epoch 120 --model_path the_path_to_save_model --train_record train_record_name
# test mmgr on downstream task (pop task), revise the pretrained-ckpt path in the python file in the first step
python res-pop.py

# BibTeX
```bash
@article{bai2023geographic,
  title={Geographic mapping with unsupervised multi-modal representation learning from VHR images and POIs},
  author={Bai, Lubin and Huang, Weiming and Zhang, Xiuyuan and Du, Shihong and Cong, Gao and Wang, Haoyu and Liu, Bo},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={201},
  pages={193--208},
  year={2023},
  publisher={Elsevier}
}
