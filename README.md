# Synth-by-Reg (SbR): Contrastive learning for synthesis-based registration of paired images

This repository contains code for intermodality registration of paired images (e.g., from the same subject). The method is synthesis based using two different 
losses: _(i)_ a registration loss for image translation at the image level that capitalises on a pre-trained intramodality registration network and _(ii)_ a structure preserving
constraint  based  on  contrastive  learning. We apply this method to the registration of histological sections to MRI slices, a key step in 3D histology reconstruction.

### Code structure
- **data** <br />
  Contains necessary data for the Allen and BigBrain datasets (reported in the paper [1])
  
- **database** <br />
  Contains I/O code for loading the datasets 
  
- **scripts** <br />
  Contains different scripts to train networks. Each folder contains a dedicated configuration file.
  
- **src** <br />
  Source code containing layer, models, losses and data loaders.
  

### Requirements:
**Python** <br />
The code run on python v3.8.5 and several external libraries listed under requirements.txt


### Run the code
- **Set-up configuration files** 
  - _setup.py_: specify data and results directory. Currently pointing to ./data and ./results. 
  
- **(Optional) Train intramodality registraion networks** 
  - _scripts/Registration/*/train.py_: train intramodality registration networks with desired parameters in configFiles from the same directory and from command line. Pre-trained registration networks are available in the results folder for both the Allen and BigBrain datasets.

- **Train intermodality registraion networks**: when using other intramodality registration networks than the ones provided, need to specify the new path in the corresponding configuration files.
  - _scripts/InfoNCE/*/train_noGAN.py_: train SbR method with parameters specified either in the configFile or from the command line. When specifygin --l_nce 0, the SbR-N is used.
  - _scripts/InfoNCE/*/train.py_: train the SbR-G extension method with parameters specified either in the configFile or from the command line. 
  - _scripts/CycleGAN/*/train.py_: train the CycleGAN baseline method, using the approach in [2] together with our registration loss.
  - _scripts/RoT/*/train.py_: train the RoT baseline method [3]

## Code updates

02 August 2021:
- Initial commit

## Citation
[1] https://arxiv.org/abs/2107.14449

## References
[2] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in IEEE International Conference on Computer Vision (ICCV), 2017. 
[3] Arar, M., Ginger, Y., Danon, D., Bermano, A. H., & Cohen-Or, D. (2020). Unsupervised multi-modal image registration via geometry preserving image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 13410-13419).
