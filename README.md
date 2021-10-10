# Coarse-Fine Networks for Temporal Activity Detection in Videos

This repository contains the official PyTorch implementation for our CVPR2021 paper titled "[Coarse-Fine Networks for Temporal Activity Detection in Videos](https://arxiv.org/abs/2103.01302)".

### Introduction

<img src="./figs/intro.png" width="600">

In this work, we introduce **Coarse-Fine Networks**, a two-stream architecture which benefits from different abstractions of temporal resolution to learn better video representations for long-term motion. Traditional Video models process inputs at one (or few) fixed temporal resolution without any dynamic frame selection. However, we argue that, processing multiple temporal resolutions of the input and doing so dynamically by learning to estimate the importance of each frame can largely improve video representations, specially in the domain of temporal activity localization. To this end, we propose (1) **Grid Pool**, a learned temporal downsampling layer to extract coarse features, and, (2) **Multi-stage Fusion**, a spatio-temporal attention mechanism to fuse a fine-grained context with the coarse features. We show that our method can outperform the state-of-the-arts for action detection in public datasets including Charades with a significantly reduced compute and memory footprint.

<img src="./figs/contrib.png" width="800">

The proposed **Grid Pool** operation learns to sample important temporal locations, creating a more meaningful coarse representation in contrast to  conventional temporal downsampling/pooling. The idea is to predict confidence scores for temporal regions at a given (output) temporal resolution, and sample based on these scores, i.e., sampling at a higher rate (or a lower sampling duration) in the regions with high confidence. We use a CDF based sampling strategy, which is similar to inverse transform sampling [(here)](https://en.wikipedia.org/wiki/Inverse_transform_sampling). Our sampling is in fact a trilinear interpolation, which makes the whole process differentiable and end-to-end trainable in turn. This is paired with a **Grid Unpool** operation, which inverts the pooling opration for accurate temporal localization in frame-wise prediction tasks such as activity detection.

Our **Multi-stage Fusion** consists of three stages: (1) filtering the fine features, (2) temporally aligning the two streams and (3) fusing multiple abstractions. We use a self-attention mask to filter which fine information should be passed through to the Coarse stream. Our two streams benefit from looking at different temporal ranges (generally the Fine stream has more range). To better fuse these representations, we need to align them temporally. We use a set of Gaussians (defined by scale and location hyperparameters) centered at coarse frame locations to aggregate fine features, weighted based on relative temporal locations. Such aligned fine features coming from multiple absrtaction levels (depths) get concatenated to predict scale and shift features for the Coarse stream.

### Results

<img src="./figs/complexity.png" width="800">

Our method achieves a new state-of-the-art on [Charades](http://vuchallenge.org/charades.html) dataset and shows competitive results on [Multi-THUMOS](http://crcv.ucf.edu/THUMOS14/download.html) dataset. This comes at a significant improvement in efficiency (more than one order of magnitude) and without any additional input modalities such as optical flow or object detections, in contrast to previous state-of-the-art methods.

### Dependencies

- Python 3.7.6
- PyTorch 1.7.0 (built from source, with [this fix](https://github.com/pytorch/pytorch/pull/40801)). This issue is fixed in PyTorch >= 1.9 releases.
- torchvision 0.8.0 (built from source)
- accimage 0.1.1
- pkbar 0.5

### Quick Start

Edit the Dataset directories to fit yours, and,

- Use `python train_fine.py -gpu {gpu_num}` for training the Fine stream.
- Use `python extract_fineFEAT.py -gpu {gpu_num}` for extracting pre-trained features from Fine stream.
- Use `python train_coarse_fineFEAT.py -gpu 0,1` for training the complete Coarse-Fine Network.

Please note that we pre-extract the fine features, and use them to train the complete Coarse-Fine Network to reduce the burden on GPUs. One can train the two streams together to achieve a better performance given enough available compute. Our pre-trained models and training logs are available. Our final reported numbers on Charades are for the `Charades_v1_localize` evaluation setting as done in all the previous work. Here, predictions are made for 25 uniformly sampled frames per video instead of densely predicting for every frame. Use [this](http://vuchallenge.org/vu17_charades.zip) evaluation script to reproduce our reported numbers.

**Check out our implementation [X3D-Multigrid](https://github.com/kkahatapitiya/X3D-Multigrid) to understand more about the backbone X3D architecture and its efficient training strategy Multigrid.**

### Reference

If you find this useful, please consider citing our work:
```
@inproceedings{kahatapitiya2021coarse,
  title={Coarse-Fine Networks for Temporal Activity Detection in Videos},
  author={Kahatapitiya, Kumara and Ryoo, Michael S},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

### Acknowledgements

I would like to thank the original authors of X3D [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.pdf) and Multigrid training [[CVPR2020]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_A_Multigrid_Method_for_Efficiently_Training_Video_Models_CVPR_2020_paper.pdf) for their inspiring work.
