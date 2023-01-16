## Semi-Supervised Learning for Low-light Image Restoration through Quality Assisted Pseudo-Labeling (WACV'2023)

[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Malik_Semi-Supervised_Learning_for_Low-Light_Image_Restoration_Through_Quality_Assisted_Pseudo-Labeling_WACV_2023_paper.pdf) &emsp; [Poster](https://drive.google.com/file/d/1g4kNQgEk1LA4C_RE26xm63a0hLpTrp1A/view?usp=sharing) &emsp; [Video](https://drive.google.com/file/d/1nk9b5i39CvIoioYqj5hDSu80Xz3eCOgp/view?usp=sharing)

### Abstract

Convolutional neural networks have been successful in restoring images captured under poor illumination conditions. Nevertheless, such approaches require a large number of paired low-light and ground truth images for training. Thus, we study the problem of semi-supervised learning for low-light image restoration when limited low-light images have ground truth labels. Our main contributions in this work are twofold. We first deploy an ensemble of low-light restoration networks to restore the unlabeled images and generate a set of potential pseudo-labels. We model the contrast distortions in the labeled set to generate different sets of training data and create the ensemble of networks. We then design a contrastive self-supervised learning based image quality measure to obtain the pseudo-label among the images restored by the ensemble. We show that training the restoration network with the pseudo-labels allows us to achieve excellent restoration performance even with very few labeled pairs. We conduct extensive experiments on three popular low-light image restoration datasets to show the superior performance of our semi-supervised low-light image restoration compared to other approaches.

#### If you find the resource useful, please cite the following :- )
