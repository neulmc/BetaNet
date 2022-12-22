# BetaNet
Created by Mingchun Li & Dali Chen

### Introduction:

we propose a new head function based on Beta distribution for boundary detection tasks. 
Different from directly learning the probability in Bernoulli distribution, 
Beta head function can introduce more abundant information. 
It can be viewed as the distribution of the parameters of the former, 
and capture the distribution uncertainty for binary classification problem. 
To this end, we combine the classical maximum likelihood and knowledge distillation 
as the loss function to train deep network with Beta head function. 
Moreover, we propose an efficient data augmentation process to expand available datasets 
with these non-deterministic labels. 
In implementation, our Beta head function is lightweight and not limited to a specific model. 
It can be seamlessly employed in many existing boundary detection networks without manually 
modifying the backbone. 
After introducing our Beta head function, the performances of three well-known boundary 
detection networks (HED, RCF, PiDiNet) have been obviously improved. 
Multiple experiments have proved the effectiveness of our method and the advantage of 
introducing uncertainty by the proposed Beta head function.



### Prerequisites

- pytorch >= 1.7.1(Our code is based on the 1.7.1)
- numpy >= 1.11.0

### Train and Evaluation
1. Clone this repository to local

2. Download the raw BSDS dataset (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) or directly use the images we extracted in this repository.

3. For these non-deterministic labels, build augmented data according to adaptive threshold and recurrent fusion strategy.

4. Use the pretrained model (HED, RCF, PiDiNet https://pan.baidu.com/s/1uGxu0h0p2hnQ8Fv7jD9NpA Code：leyo ) to predict the training images and get soft labels for knowledge distillation.

5. Train our proposed BetaNet for learning Beta distribution.

We have released the final prediction and evaluation results, which can be downloaded at the following link:
https://pan.baidu.com/s/15CigFVsC8huebj1zUAm9ug  Code：w5c1

### Preprocessing Dataset 
We recommend getting the augmented dataset from the raw dataset integrated in our repository (about 50M), 
and use the teacher model (about 120M) to obtain the soft label of the augmented dataset. 
However, we also provide compiled augmented datasets, as shown below

```
BetaBSDS/
├── train/ (https://pan.baidu.com/s/1jAO0HCgs7fbiYiHQmus6QQ Code: 7i02)
│   ├── aug_data/ 
│   ├── aug_data_scale_0.5/
│   ├── aug_data_scale_1.5/
│   ├── aug_gt(beta)/
│   ├── aug_gt_scale_0.5(beta)/
│   └── aug_gt_scale_1.5(beta)/
├── distill_hed/ (https://pan.baidu.com/s/1H07GPSEUT_RaGCSGMEoemg Code: 7pvo)
│   ├── aug_data/
│   ├── aug_data_scale_0.5/
│   └── aug_data_scale_1.5/
├── distill_rcf/ (https://pan.baidu.com/s/1Duk3JE9tXmLYBk1XeVnELA Code: w6fd)
│   ├── aug_data/
│   ├── aug_data_scale_0.5/
│   └── aug_data_scale_1.5/
├── distill_pidinet/ (https://pan.baidu.com/s/1SnwFV-xQHKQqmIOScZCXCg Code: ptti)
│   ├── aug_data/
│   ├── aug_data_scale_0.5/
│   └── aug_data_scale_1.5/
├── test/
├── train_pair.lst
└── test.lst
```

### Results on BSDS Dataset
| Method |ODS F-score |OIS F-score |
|:-----|:-----:|:-----:| 
| HED-Beta (Ours)| 0.795 | 0.816 |
| HED [1]| 0.788 | 0.808  |
| RCF-Beta (Ours)| 0.803 | 0.822 |
| RCF pytorch Version| 0.796 | 0.814 |
| RCF [2]| 0.798 | 0.815  |
| PiDiNet-Beta (Ours)| 0.800 | 0.814 |
| PiDiNet [3]| 0.789 | 0.803  |

### Final models
This is the final model in our paper. We used this model to evaluate. You can download by:
https://pan.baidu.com/s/1nbBG_tcp22nWuhFVrEOVqw  Code：j5fl 

### Notes
If we want to train BetaRCF or BataHED, they need to be initialized by VGG16. 
So we should download the pre-trained [VGG16](https://drive.google.com/file/d/1lUhPKKj-BSOH7yQL0mOIavvrUbjydPp5/view?usp=sharing) model before.

### Acknowledgment
Part of our code comes from [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf), [Pidinet Repository](https://github.com/zhuoinoulu/pidinet). We are very grateful for these excellent works.
In order to reproduce the existing excellent models and serve as the teacher models, 
the three available and effective repositories implemented by PyTorch are [HED Repository](https://github.com/xwjabc/hed), 
[RCF Repository](https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated), and [Pidinet Repository](https://github.com/zhuoinoulu/pidinet).

### References
[1] <a href="https://arxiv.org/abs/1504.06375v2">Holistically-nested edge detection</a>

[2] <a href="https://arxiv.org/abs/1612.02103v3">Richer convolutional features for edge detection</a>

[3] <a href="https://arxiv.org/abs/2108.07009">Pixel difference networks for efficient edge detection</a>
