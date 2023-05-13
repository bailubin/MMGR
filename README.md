# MMGR
Official implementation of the paper "Geographic mapping with unsupervised multi-modal representation learning from VHR images and POIs", a multimodal self-supervised contrastive learning framwork using very high resolution (VHR) images and POIs. The learned representations can be readily applied to various geographic mapping tasks, e.g., urban function region classification, population density distribution and GDP distribution mapping. Inaddition, MMGR is a competent pre-training method to help image encoders understand multi-modal geographic information, which can be bring some inspiration to the design of foundation model in remote sensing field.
# requirements
torch
torchvision
torch_geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
