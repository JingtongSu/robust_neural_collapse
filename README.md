# On the Robustness of Neural Collapse and the Neural Collapse of Robustness
This repository contains the code for reproducing the results in the following paper:

On the Robustness of Neural Collapse and the Neural Collapse of Robustness. [[link]](https://openreview.net/forum?id=OyXS4ZIqd3)

[Jingtong Su](https://cims.nyu.edu/~js12196/), [Ya Shi Zhang](https://yashizhang.github.io/), [Nikolaos Tsilivis](https://cims.nyu.edu/~nt2231/page.html), and [Julia Kempe](https://cims.nyu.edu/~kempe/).

## Dependencies

pytorch v1.12.0, scipy v1.10.1, numpy v1.23.5, torchvision v0.13.0

## Introduction of Code Files

The code files are used to collect accuracy, loss and Neural Collapse results for standardly-trained (ST), adversarially-trained (AT), and TRADES-trained (TRADES) networks.

vgg.py and preactresnet.py are model files that define the PreActResNet18 and VGG11 networks. For ImageNette, since the input size is 160x160, we need to modify the corresponding network structure to fit this change. The adopted models can be found in vgg_imagenet.py and preactresnet_imagenet.py.

train.py is used to train a network for 400 epochs. For AT and TRADES, we must complete NC evaluation during training because we need to track the NC quantities on exactly the data used in training. Take CIFAR-10, $\ell_\infty$ adversary, VGG11 as an example, the commands for ST, AT, and TRADES are as follows: (we used seed = 1, 2, and 3 in our experiments)

ST:

```
python ./train.py --dataset=cifar --model=vgg11bn --seed 1 --norm l_inf --train-mode std_train --lr-max=0.1 --epsilon=8 --pgd-alpha=2 --attack-iters=10 --epochs 400
```

AT:

```
python ./train.py --dataset=cifar --model=vgg11bn --seed 1 --norm l_inf --train-mode pgd_train --lr-max=0.1 --epsilon=8 --pgd-alpha=2 --attack-iters=10 --epochs 400 --neural-collapse
```

TRADES:

```
python ./train.py --dataset=cifar --model=vgg11bn --seed 1 --norm l_inf --train-mode pgd_train_TRADES --lr-max=0.1 --epsilon=8 --pgd-alpha=2 --attack-iters=10 --epochs 400 --neural-collapse
```

eval.py is used to evaluate a given series of models (400 in total). This file is used to collect accuracy, loss, and NC statistics. For AT, since we have already collected the clean/perturbed data and classifier statistics, here we only need to track the Gaussian baseline shown in our paper. For TRADES, we don't need this file because we found no simplices associated with it, so we did not conduct a Gaussian baseline with it. With ST, we need to track clean/perturbed data, the Gaussian baseline, and the targeted attack we propose. Still, take CIFAR-10, $\ell_\infty$ adversary, VGG11 as an example, the commands for ST, AT, and TRADES are as follows:

ST:
```
python ./eval.py --dataset=cifar --model=vgg11bn --seed 1 --norm l_inf --train-mode std_train --lr-max=0.1 --epsilon=8 --pgd-alpha=2 --attack-iters=10 --epochs 400 --neural-collapse --neural-collapse-gaussian --neural-collapse-targeted
```

AT:
```
python ./eval.py --dataset=cifar --model=vgg11bn --seed 1 --norm l_inf --train-mode pgd_train --lr-max=0.1 --epsilon=8 --pgd-alpha=2 --attack-iters=10 --epochs 400  --neural-collapse-gaussian
```

## Citation
If you use our code in your research, please cite:
~~~
@article{su2023robustness,
  title={On the Robustness of Neural Collapse and the Neural Collapse of Robustness},
  author={Su, Jingtong and Zhang, Ya Shi and Tsilivis, Nikolaos and Kempe, Julia},
  journal={arXiv preprint arXiv:2311.07444},
  year={2023}
}
