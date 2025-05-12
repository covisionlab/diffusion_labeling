# Bounding Box-Guided Diffusion for Industrial Image Synthesis

This repository contains the official implementation of the paper:
["Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map"](https://arxiv.org/abs/2505.03623) accepted at Synthetic Data for Computer Vision Workshop - CVPR 2025


## 📌 Overview

This project introduces a diffusion-based generative framework guided by bounding boxes to synthesize high-quality industrial images along with corresponding segmentation maps. The method is designed to support precise localization, multi-part control, and mask generation, facilitating dataset creation for downstream tasks like defect detection and segmentation.

## 🖇️ Setup
Clone the repository and install the necessary dependencies:
```
git clone https://github.com/covisionlab/diffusion_labeling
cd diffusion_labeling
python3 -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## 🚀 Usage

The repo is composed by three modules. That should be run consequentely:

1. `preprocess`: this module preprocess the original wood dataset which can be found here [https://zenodo.org/records/4694695#.YkWqTX9Bzmg](https://zenodo.org/records/4694695#.YkWqTX9Bzmg). Read the `preprocess/README.md` for more information.

2. `generation`: this module run the diffusion pipeline described in the paper, and generates the synthetic data which will be used in 3. Read the `generation/README.md` for more information.

3. `segmentation`: this is the segmentation module which should be run at the end of the pipeline to retrieve the metrics ebr, fid, sae, f1. Read the `segmentation/README.md` for more information.

We provide inside `data/splits` the official splits of the dataset we used to train our diffusion. So you can replicate the results.

## 📄 Citation

If you use this code in your research, please cite:

```
@inproceedings{
    simoni2025bounding,
    title={Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Maps},
    author={Alessandro Simoni and Francesco Pelosin},
    booktitle={Synthetic Data for Computer Vision Workshop @ CVPR 2025},
    year={2025}
}
```
