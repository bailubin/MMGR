# MMGR
Official implementation of the paper "Geographic mapping with unsupervised multi-modal representation learning from VHR images and POIs", a multimodal self-supervised contrastive learning framwork using very high resolution (VHR) images and POIs. 
> **Abstract.**
> Most supervised geographic mapping methods with very-high-resolution (VHR) images are designed for a specific task, leading to high label-dependency and inadequate task-generality. Additionally, the lack of socio-economic information in VHR images limits their applicability to social/human-related geographic studies. To resolve these two issues, we propose an unsupervised multi-modal geographic representation learning framework (MMGR) using both VHR images and points-of-interest (POIs), to learn representations (regional vector embeddings) carrying both the physical and socio-economic properties of the geographies. In MMGR, we employ an intra-modal and an inter-modal contrastive learning module, in which the former deeply mines visual features by contrasting different VHR image augmentations, while the latter fuses physical and socio-economic features by contrasting VHR image and POI features. Extensive experiments are performed in two study areas (Shanghai and Wuhan in China) and three relevant while distinctive geographic mapping tasks (i.e., mapping urban functional distributions, population density, and gross domestic product), to verify the superiority of MMGR. The results demonstrate that the proposed MMGR considerably outperforms seven competitive baselines in all three tasks, which indicates its effectiveness in fusing VHR images and POIs for multiple geographic mapping tasks. Furthermore, MMGR is a competent pre-training method to help image encoders understand multi-modal geographic information, and it can be further strengthened by fine-tuning even with a few labeled samples.
# requirements
torch
torchvision
torch_geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
