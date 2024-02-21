import argparse
import logging
import sys
import time
import math
import matplotlib.pyplot as plt
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.sparse.linalg import svds
from scipy.linalg import pinv

import os

from vgg import VGG, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from preactresnet import PreActResNet18
from preactresnet_imagenet import preact_resnet18
from vgg_imagenet import vgg11_bn_imnet, vgg11_bn_pool
from torch import Tensor

logging.getLogger('matplotlib.font_manager').disabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_list_as_npy(mylist, base_dir, name):
    mylist = np.array(mylist)
    np.save(os.path.join(base_dir, name), mylist)
    
def normalize(X): # Reimplement
    return (X - mu) / std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def coherence(V, num_classes=10):
    G = V.T @ V 
    G += torch.ones((num_classes,num_classes), device=device) / (num_classes-1)
    G -= torch.diag(torch.diag(G))
    return torch.norm(G,1).item() / (num_classes * (num_classes-1))


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def attack_trades(model,
                x_natural,
                y,
                epsilon=8/255.,
                step_size=2/255.,
                perturb_steps=10,
                beta=6.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(normalize(x_adv)), dim=1),
                                       F.softmax(model(normalize(x_natural)), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = torch.optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(normalize(adv)), dim=1),
                                           F.softmax(model(normalize(x_natural)), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv - x_natural


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, targeted=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            if targeted:
                loss = -loss
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def neural_collapse(model, device, criterion, classifier, clean_dataloader, perturbed_dataloader, num_classes=10, standard_training=False):
    model.eval() 
    N = [0 for _ in range(num_classes)]
    mean = [0 for _ in range(num_classes)]
    Sw = 0
    loss = 0
    net_correct = 0
    zeros_ready = None
    NCC_match_net = 0
    dataloader = {"Mean": clean_dataloader, "Cov": perturbed_dataloader}
    with torch.no_grad():
        for computation in["Mean", "Cov"]:
            for batch_idx, (data, target) in enumerate(dataloader[computation]):
                data, target = data.to(device), target.to(device)
                output, h = model(data, return_activations=True)
                
                zeros_ready  = torch.zeros_like(torch.sum(h, dim=0))

                if computation == "Mean":
                    loss += criterion(output, target).item()
                
                for c in range(num_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]

                    if len(idxs) == 0:
                        continue
                    
                    h_c = h[idxs, :]

                    if computation == "Mean":
                        mean[c] += torch.sum(h_c, dim=0)
                        N[c] += h_c.shape[0]
                    elif computation == "Cov":
                        z = h_c - mean[c].unsqueeze(0)
                        cov = torch.matmul(
                            z.unsqueeze(-1), z.unsqueeze(1) 
                        ) 
                        Sw += torch.sum(cov, dim=0)

                        net_pred = torch.argmax(output[idxs, :], dim=1)
                        net_correct += sum(net_pred == target[idxs]).item()

                        NCC_scores = torch.stack(
                            [
                                torch.norm(h_c[i, :] - M.T, dim=1)
                                for i in range(h_c.shape[0])
                            ]
                        )
                        NCC_pred = torch.argmin(NCC_scores, dim=1)
                        NCC_match_net += sum(NCC_pred == net_pred).item()
            if computation == "Mean":
                for c in range(num_classes):
                    if N[c] > 0:
                        mean[c] /= N[c]
                    else:
                        mean[c] = zeros_ready
                M = torch.stack(mean).T 
                loss /= sum(N)
            elif computation == "Cov":
                Sw /= sum(N)
        muG = 0
        for c in range(num_classes):
            muG += mean[c].unsqueeze(0).T * N[c]
        muG /= sum(N)
#         muG = torch.mean(M, dim=1, keepdim=True)
        M_ = M - muG
        Sb = torch.matmul(M_, M_.T) / num_classes
        try:
            W = classifier[0].weight
        except:
            W = classifier.weight
        M_norms = torch.norm(M_, dim=0)
        W_norms = torch.norm(W.T, dim=0)
        
        Sw = Sw.cpu().numpy()
        Sb = Sb.cpu().numpy()

        inv_Sb = pinv(Sb)

        normalized_M = M_ / torch.norm(M_, "fro")
        normalized_W = W.T / torch.norm(W.T, "fro")

        nc1 = np.trace(Sw @ inv_Sb)
        nc2_norm_M = (torch.std(M_norms) / torch.mean(M_norms)).item()
        nc2_norm_W = (torch.std(W_norms) / torch.mean(W_norms)).item()
        nc2_angle_M = coherence(M_ / M_norms, num_classes=num_classes)
        nc2_angle_W = coherence(W.T / W_norms, num_classes=num_classes)
        nc3 = (torch.norm(normalized_W - normalized_M) ** 2).item()
        nc4 = 1 - NCC_match_net / sum(N)
        
        # for each pairwise column vector in M_, compute their angle with each other (NC2)
        angles_activations = []
        angles_weights = []
        for i in range(M_.shape[1]):
            for j in range(i+1, M_.shape[1]):
                angles_activations.append((torch.dot(M_[:, i], M_[:, j]) / (torch.norm(M_[:, i]) * torch.norm(M_[:, j]))).clone().cpu().numpy())
                angles_weights.append((torch.dot(W.T[:, i], W.T[:, j]) / (torch.norm(W.T[:, i]) * torch.norm(W.T[:, j]))).clone().cpu().numpy())
        # for each column in M_, compute its L2 norm (NC2)
        norms_activations = []
        norms_weights = []
        for i in range(M_.shape[1]):
            norms_activations.append(torch.norm(M_[:, i]).clone().cpu().numpy())
            norms_weights.append((torch.norm(W.T[:, i])).clone().cpu().numpy())
        # copy M_ as numpy array and return them for NC3 analysis
        #M_ = M_.clone().cpu().numpy()
    return nc1, nc2_norm_M, nc2_norm_W, nc2_angle_M, nc2_angle_W, nc3, nc4 , (norms_activations, angles_activations), (norms_weights, angles_weights), M #M_

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--model', default='VGG11')
    parser.add_argument('--train-mode', default='std_train')
    parser.add_argument('--l2', default=0, type=float) # DONT USE
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--standard-training', action='store_true')
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=0.1, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='mnist_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=1, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=500, type=int) # TO SAVE SPACE
    parser.add_argument('--neural-collapse', action='store_true')
    parser.add_argument('--neural-collapse-gaussian', action='store_true')
    parser.add_argument('--neural-collapse-targeted', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    print(args)
    train_method = args.train_mode
    assert train_method in ['std_train', 'pgd_train', 'pgd_train_TRADES'], 'Please Specify the correct Train Method!'

    save_path = f'./models/{args.dataset}/{args.model}/{args.norm}/eps{args.epsilon}/{train_method}/seed{args.seed}'
    
    load_path = f'./models/{args.dataset}/{args.model}/{args.norm}/eps{args.epsilon}/{train_method}/seed{args.seed}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_workers = 1
    if (args.dataset == 'mnist'):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.MNIST(
            args.data_dir,
            train=True,
            transform=train_transform,
            download=True
        )
        # keep 5000 from each class for balance
        train_dataset = torch.utils.data.ConcatDataset([
            torch.utils.data.Subset(train_dataset, np.random.choice(np.where(np.array(train_dataset.targets) == i)[0], 5000, replace=False))
            for i in range(10)
        ])
        test_dataset = datasets.MNIST(
            args.data_dir,
            train=False,
            transform=test_transform,
            download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )
    elif (args.dataset == 'cifar'):
        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(
        args.data_dir,
        train=True,
        transform=train_transform,
        download=True
        )
        test_dataset = datasets.CIFAR10(
            args.data_dir,
            train=False,
            transform=test_transform,
            download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )
    elif (args.dataset == 'cifar100'):
        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR100(
        args.data_dir,
        train=True,
        transform=train_transform,
        download=True
        )
        test_dataset = datasets.CIFAR100(
            args.data_dir,
            train=False,
            transform=test_transform,
            download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )
    elif (args.dataset == 'imnet'):
        train_transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=test_transform
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )                
    else:
        raise NotImplementedError

    global mu, std 
    if args.dataset == 'cifar':
        mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
        std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()
        epsilon = args.epsilon / 255.
        pgd_alpha = args.pgd_alpha / 255. 
        in_channels = 3
        imsize = 32
        num_classes = 10
    elif args.dataset == 'mnist':
        mu = torch.tensor((0.1307,)).view(1,1,1).cuda()
        std = torch.tensor((0.3081,)).view(1,1,1).cuda()
        epsilon = args.epsilon
        pgd_alpha = args.pgd_alpha
        in_channels = 1
        imsize = 28
        num_classes = 10
    elif args.dataset == 'cifar100':
        mu = torch.tensor((0.5071, 0.4865, 0.4409)).view(3,1,1).cuda()
        std = torch.tensor((0.2673, 0.2564, 0.2762)).view(3,1,1).cuda()
        epsilon = args.epsilon / 255.
        pgd_alpha = args.pgd_alpha / 255. 
        in_channels = 3
        imsize = 32
        num_classes = 100
    elif args.dataset == 'imnet':
        mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
        std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
        epsilon = args.epsilon / 255.
        pgd_alpha = args.pgd_alpha / 255. 
        in_channels = 3
        imsize = 160
        num_classes = 10

    if args.model == 'vgg11bn':
        if args.dataset == 'imnet':
            model = vgg11_bn_pool(num_classes = num_classes)
            classifier = model.classifier
        else:
            model = vgg11_bn(in_channels=in_channels, num_classes=num_classes)
            classifier = model.classifier
    elif args.model == 'PreActResNet18':
        if args.dataset == 'imnet':
            model = preact_resnet18(num_classes=num_classes)
            classifier = model.fc
        else:
            model = PreActResNet18(num_classes=num_classes, in_channels=in_channels)
            classifier = model.linear 
    else:
        raise ValueError('Model not supported')

    model = model.cuda()
    
#     names.append('ori')
#     print("Names:")
#     print(names)
    
    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                    {'params': no_decay, 'weight_decay': 0.0}]
    else:
        params = model.parameters()

    
    criterion = nn.CrossEntropyLoss()
        
    train_loss_plot, train_acc_plot, train_robust_loss_plot, train_robust_acc_plot = [], [], [], []
    test_loss_plot, test_acc_plot, test_robust_loss_plot, test_robust_acc_plot = [], [], [], []
    nc1_plot, nc2_norm_M_plot, nc2_norm_W_plot, nc2_angle_M_plot, nc2_angle_W_plot, nc3_plot, nc4_plot = [], [], [], [], [], [], []
    robust_nc1_plot, robust_nc2_norm_M_plot, robust_nc2_norm_W_plot, robust_nc2_angle_M_plot, robust_nc2_angle_W_plot, robust_nc3_plot, robust_nc4_plot = [], [], [], [], [], [], []
    reverse_nc1_plot, reverse_nc2_norm_M_plot, reverse_nc2_norm_W_plot, reverse_nc2_angle_M_plot, reverse_nc2_angle_W_plot, reverse_nc3_plot, reverse_nc4_plot = [], [], [], [], [], [], []
    perturb_nc1_plot, perturb_nc2_norm_M_plot, perturb_nc2_norm_W_plot, perturb_nc2_angle_M_plot, perturb_nc2_angle_W_plot, perturb_nc3_plot, perturb_nc4_plot = [], [], [], [], [], [], []
    
    test_nc1_plot, test_nc2_norm_M_plot, test_nc2_norm_W_plot, test_nc2_angle_M_plot, test_nc2_angle_W_plot, test_nc3_plot, test_nc4_plot = [], [], [], [], [], [], []
    test_robust_nc1_plot, test_robust_nc2_norm_M_plot, test_robust_nc2_norm_W_plot, test_robust_nc2_angle_M_plot, test_robust_nc2_angle_W_plot, test_robust_nc3_plot, test_robust_nc4_plot = [], [], [], [], [], [], []
    test_reverse_nc1_plot, test_reverse_nc2_norm_M_plot, test_reverse_nc2_norm_W_plot, test_reverse_nc2_angle_M_plot, test_reverse_nc2_angle_W_plot, test_reverse_nc3_plot, test_reverse_nc4_plot = [], [], [], [], [], [], []
    test_perturb_nc1_plot, test_perturb_nc2_norm_M_plot, test_perturb_nc2_norm_W_plot, test_perturb_nc2_angle_M_plot, test_perturb_nc2_angle_W_plot, test_perturb_nc3_plot, test_perturb_nc4_plot = [], [], [], [], [], [], []
    
    targeted_nc1_plot, targeted_nc2_norm_M_plot, targeted_nc2_norm_W_plot, targeted_nc2_angle_M_plot, targeted_nc2_angle_W_plot, targeted_nc3_plot, targeted_nc4_plot = [], [], [], [], [], [], []

    nc1, nc2_norm_M, nc2_norm_W, nc2_angle_M, nc2_angle_W, nc3, nc4 = 0, 0, 0, 0, 0, 0, 0
    robust_nc1, robust_nc2_norm_M, robust_nc2_norm_W, robust_nc2_angle_M, robust_nc2_angle_W, robust_nc3, robust_nc4 = 0, 0, 0, 0, 0, 0, 0
    reverse_nc1, reverse_nc2_norm_M, reverse_nc2_norm_W, reverse_nc2_angle_M, reverse_nc2_angle_W, reverse_nc3, reverse_nc4 = 0, 0, 0, 0, 0, 0, 0
    perturb_nc1, perturb_nc2_norm_M, perturb_nc2_norm_W, perturb_nc2_angle_M, perturb_nc2_angle_W, perturb_nc3, perturb_nc4 = 0, 0, 0, 0, 0, 0, 0
    
    test_nc1, test_nc2_norm_M, test_nc2_norm_W, test_nc2_angle_M, test_nc2_angle_W, test_nc3, test_nc4 = 0, 0, 0, 0, 0, 0, 0
    test_robust_nc1, test_robust_nc2_norm_M, test_robust_nc2_norm_W, test_robust_nc2_angle_M, test_robust_nc2_angle_W, test_robust_nc3, test_robust_nc4 = 0, 0, 0, 0, 0, 0, 0
    test_reverse_nc1, test_reverse_nc2_norm_M, test_reverse_nc2_norm_W, test_reverse_nc2_angle_M, test_reverse_nc2_angle_W, test_reverse_nc3, test_reverse_nc4 = 0, 0, 0, 0, 0, 0, 0
    test_perturb_nc1, test_perturb_nc2_norm_M, test_perturb_nc2_norm_W, test_perturb_nc2_angle_M, test_perturb_nc2_angle_W, test_perturb_nc3, test_perturb_nc4 = 0, 0, 0, 0, 0, 0, 0
    
    targeted_nc1, targeted_nc2_norm_M, targeted_nc2_norm_W, targeted_nc2_angle_M, targeted_nc2_angle_W, targeted_nc3, targeted_nc4 = 0, 0, 0, 0, 0, 0, 0

    clean_activations_norm_plot, clean_activations_angle_plot, robust_activations_norm_plot, robust_activations_angle_plot, reverse_activations_norm_plot, reverse_activations_angle_plot, perturb_activations_norm_plot, perturb_activations_angle_plot = [], [], [], [], [], [], [], []
    
    perturbed_label_nc1_plot, perturbed_label_nc2_norm_M_plot, perturbed_label_nc2_norm_W_plot, perturbed_label_nc2_angle_M_plot, perturbed_label_nc2_angle_W_plot, perturbed_label_nc3_plot, perturbed_label_nc4_plot = [], [], [], [], [], [], []
    
    gaussian_label_nc1_plot, gaussian_label_nc2_norm_M_plot, gaussian_label_nc2_norm_W_plot, gaussian_label_nc2_angle_M_plot, gaussian_label_nc2_angle_W_plot, gaussian_label_nc3_plot, gaussian_label_nc4_plot = [], [], [], [], [], [], []
    
    linf_label_nc1_plot, linf_label_nc2_norm_M_plot, linf_label_nc2_norm_W_plot, linf_label_nc2_angle_M_plot, linf_label_nc2_angle_W_plot, linf_label_nc3_plot, linf_label_nc4_plot = [], [], [], [], [], [], []

    weights_norm_plot, weights_angle_plot = [], []
    clean_robust_dist_plot, clean_robust_angle_plot, clean_robust_dist_plot_abs, clean_robust_angle_plot_abs, muG_dist_plot, muG_angle_plot = [], [], [], [], [], []
    clean_target_dist_plot, clean_target_angle_plot, clean_target_dist_plot_abs, clean_target_angle_plot_abs, target_muG_dist_plot, target_muG_angle_plot = [], [], [], [], [], []

    
    
    for epoch in range(400):
        model.load_state_dict(torch.load(os.path.join(load_path, f'model_{epoch}.pth'))['state_dict'])
        model.cuda()
        model.eval()

        clean_train_data = []
        perturbed_train_data = []
        train_loss, train_acc, train_robust_loss, train_robust_acc, train_n = 0, 0, 0, 0, 0
        train_time = time.time()
        print(f"--- Epoch #{epoch + 1} ---")
        if args.neural_collapse:
            print("Start Attacking Training Data...")
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()

                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)

                delta = delta.detach()
                
                output = model(normalize(X))
                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                loss = criterion(output, y)
                robust_loss = criterion(robust_output, y)
                
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                train_robust_loss += robust_loss.item() * y.size(0)
                train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                
                X_perturbed = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)).detach().cpu()
                y = y.detach().cpu()
                clean_train_data.append((normalize(X).cpu(), y))
                perturbed_train_data.append((X_perturbed, y))
                

        if args.neural_collapse:        
            train_loss_plot.append(train_loss / train_n)
            train_acc_plot.append(train_acc / train_n)
            train_robust_loss_plot.append(train_robust_loss / train_n)
            train_robust_acc_plot.append(train_robust_acc / train_n)

        # Per Epoch Neural Collapse Metric Calculation
        NC_time = time.time()
        print("Time Elapsed: %f" % (NC_time - train_time))
        print("Start NC Eval...")
        if args.neural_collapse:
            nc1, nc2_norm_M, nc2_norm_W, nc2_angle_M, nc2_angle_W, nc3, nc4, clean_activations, weights, clean_M = neural_collapse(model, device, nn.CrossEntropyLoss(reduction="sum"), classifier, clean_train_data, clean_train_data, num_classes=num_classes, standard_training=args.standard_training)
            nc1_plot.append(nc1)
            nc2_norm_M_plot.append(nc2_norm_M)
            nc2_norm_W_plot.append(nc2_norm_W)
            nc2_angle_M_plot.append(nc2_angle_M)
            nc2_angle_W_plot.append(nc2_angle_W)
            nc3_plot.append(nc3)
            nc4_plot.append(nc4)

            robust_nc1, robust_nc2_norm_M, robust_nc2_norm_W, robust_nc2_angle_M, robust_nc2_angle_W, robust_nc3, robust_nc4, robust_activations, _, robust_M = neural_collapse(model, device, nn.CrossEntropyLoss(reduction="sum"), classifier, perturbed_train_data, perturbed_train_data, num_classes=num_classes, standard_training=args.standard_training)
            robust_nc1_plot.append(robust_nc1)
            robust_nc2_norm_M_plot.append(robust_nc2_norm_M)
            robust_nc2_norm_W_plot.append(robust_nc2_norm_W)
            robust_nc2_angle_M_plot.append(robust_nc2_angle_M)
            robust_nc2_angle_W_plot.append(robust_nc2_angle_W)
            robust_nc3_plot.append(robust_nc3)
            robust_nc4_plot.append(robust_nc4)

        if args.neural_collapse_gaussian:
            gaussian_data = []
            linf_data = []
            with torch.no_grad():
                for i, (X, y) in enumerate(train_loader):
                    X, y = X.cuda(), y.cuda()
                    rand_gaussian = torch.randn(X.shape).cuda().detach()
                    X_linf = X.detach() + 8/255. * torch.sign(rand_gaussian)
#                     scaled_rand_gaussian = rand_gaussian.view(rand_gaussian.size(0),-1).renorm(p=2, dim=0, maxnorm=128/255).view_as(rand_gaussian)
                    scaled_rand_gaussian = rand_gaussian * (8 / 255.)
                    X_gaussian = X.detach() + scaled_rand_gaussian
                    gaussian_data.append((normalize(torch.clamp(X_gaussian, min=lower_limit, max=upper_limit)), y))
                    linf_data.append((normalize(torch.clamp(X_linf, min=lower_limit, max=upper_limit)), y))

            gaussian_label_nc1, gaussian_label_nc2_norm_M, gaussian_label_nc2_norm_W, gaussian_label_nc2_angle_M, gaussian_label_nc2_angle_W, gaussian_label_nc3, gaussian_label_nc4, _, _, _ = neural_collapse(model, device, nn.CrossEntropyLoss(reduction="sum"), classifier, gaussian_data, gaussian_data, num_classes=num_classes)
            gaussian_label_nc1_plot.append(gaussian_label_nc1)
            gaussian_label_nc2_norm_M_plot.append(gaussian_label_nc2_norm_M)
            gaussian_label_nc2_norm_W_plot.append(gaussian_label_nc2_norm_W)
            gaussian_label_nc2_angle_M_plot.append(gaussian_label_nc2_angle_M)
            gaussian_label_nc2_angle_W_plot.append(gaussian_label_nc2_angle_W)
            gaussian_label_nc3_plot.append(gaussian_label_nc3)
            gaussian_label_nc4_plot.append(gaussian_label_nc4)

            linf_label_nc1, linf_label_nc2_norm_M, linf_label_nc2_norm_W, linf_label_nc2_angle_M, linf_label_nc2_angle_W, linf_label_nc3, linf_label_nc4, _, _, _ = neural_collapse(model, device, nn.CrossEntropyLoss(reduction="sum"), classifier, linf_data, linf_data, num_classes=num_classes)
            linf_label_nc1_plot.append(linf_label_nc1)
            linf_label_nc2_norm_M_plot.append(linf_label_nc2_norm_M)
            linf_label_nc2_norm_W_plot.append(linf_label_nc2_norm_W)
            linf_label_nc2_angle_M_plot.append(linf_label_nc2_angle_M)
            linf_label_nc2_angle_W_plot.append(linf_label_nc2_angle_W)
            linf_label_nc3_plot.append(linf_label_nc3)
            linf_label_nc4_plot.append(linf_label_nc4)

        if args.neural_collapse_targeted:
            targeted_train_data = []
            for i, (X, y) in enumerate(train_loader):
                X, y  = X.cuda(), y.cuda()
                y = (y + 1) % num_classes # Targeted attack
                # we always use PGD for attack purposes, with standard training
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, targeted=True)
                delta = delta.detach()
                X_perturbed = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)).cpu()
                y = y.cpu()
                targeted_train_data.append((X_perturbed, y))
            targeted_nc1, targeted_nc2_norm_M, targeted_nc2_norm_W, targeted_nc2_angle_M, targeted_nc2_angle_W, targeted_nc3, targeted_nc4, _, _, target_M = neural_collapse(model, device, nn.CrossEntropyLoss(reduction="sum"), classifier, targeted_train_data, targeted_train_data, num_classes=num_classes, standard_training=args.standard_training)
            targeted_nc1_plot.append(targeted_nc1)
            targeted_nc2_norm_M_plot.append(targeted_nc2_norm_M)
            targeted_nc2_norm_W_plot.append(targeted_nc2_norm_W)
            targeted_nc2_angle_M_plot.append(targeted_nc2_angle_M)
            targeted_nc2_angle_W_plot.append(targeted_nc2_angle_W)
            targeted_nc3_plot.append(targeted_nc3)
            targeted_nc4_plot.append(targeted_nc4)
            # For each column vector in clean_weights and robust_weights, calculate their L2 distance and angle and store them in a list 
            target_mean_act_dist = []
            target_mean_act_angle = []
            target_mean_act_dist_abs = []
            target_mean_act_angle_abs = []
            target_muG_dist = []
            target_muG_angle = []

            clean_muG = torch.mean(clean_M, dim=1, keepdim=True)
            clean_M_ = clean_M - clean_muG
            clean_M_ = clean_M_.clone().cpu().numpy()
            clean_M = clean_M.clone().cpu().numpy()
            clean_muG = clean_muG.clone().cpu().numpy()

            target_muG = torch.mean(target_M, dim=1, keepdim=True)
            target_M_ = target_M - target_muG
            target_M_ = target_M_.clone().cpu().numpy()
            target_M = target_M.clone().cpu().numpy()
            target_muG = target_muG.clone().cpu().numpy()

            clean_muG = clean_muG[:, 0]
            target_muG = target_muG[:, 0]

            for i in range(clean_M.shape[1]):
                target_mean_act_dist_abs.append(np.linalg.norm(clean_M[:,i] - target_M[:,i]))
                target_mean_act_angle_abs.append(np.arccos(np.dot(clean_M[:,i], target_M[:,i]) / (np.linalg.norm(clean_M[:,i]) * np.linalg.norm(target_M[:,i]) + 1e-10)))
                target_mean_act_dist.append(np.linalg.norm(clean_M_[:,i] - target_M_[:,i]))
                target_mean_act_angle.append(np.arccos(np.dot(clean_M_[:,i], target_M_[:,i]) / (np.linalg.norm(clean_M_[:,i]) * np.linalg.norm(target_M_[:,i]) + 1e-10)))
            target_muG_dist.append(np.linalg.norm(clean_muG - target_muG))
            target_muG_angle.append(np.arccos(np.dot(clean_muG, target_muG) / (np.linalg.norm(clean_muG) * np.linalg.norm(target_muG) + 1e-10)))
            # Append list (idx by class) of L2 distances and angles of clean mean activation and robust mean activations to the list of lists 
            clean_target_dist_plot.append(target_mean_act_dist)
            clean_target_angle_plot.append(target_mean_act_angle)
            clean_target_dist_plot_abs.append(target_mean_act_dist_abs)
            clean_target_angle_plot_abs.append(target_mean_act_angle_abs)
            target_muG_dist_plot.append(target_muG_dist)
            target_muG_angle_plot.append(target_muG_angle)

        nc_time = time.time()
        print("Time Elapsed: %f" % (nc_time - NC_time))
        print('One Epoch Done. Storing Results...')
        print()
        res_path = os.path.join(save_path, 'results')
        os.makedirs(res_path, exist_ok=True)

        fig_path = os.path.join(save_path, 'figures')
        os.makedirs(fig_path, exist_ok=True)
        if args.neural_collapse:
            # save results
            save_list_as_npy(train_loss_plot, res_path, 'train_loss')
            save_list_as_npy(train_robust_loss_plot, res_path, 'train_robust_loss')

            # save results
            save_list_as_npy(train_acc_plot, res_path, 'train_acc')
            save_list_as_npy(train_robust_acc_plot, res_path, 'train_robust_acc')

            # save results
            save_list_as_npy(nc1_plot, res_path, 'nc1')
            save_list_as_npy(robust_nc1_plot, res_path, 'robust_nc1')

            # save results
            save_list_as_npy(nc2_norm_M_plot, res_path, 'nc2_activation_norm')
            save_list_as_npy(robust_nc2_norm_M_plot, res_path, 'robust_activation_nc2_norm')
            save_list_as_npy(nc2_norm_W_plot, res_path, 'nc2_classifier_norm')
            save_list_as_npy(robust_nc2_norm_W_plot, res_path, 'robust_classifier_nc2_norm')

            # save results
            save_list_as_npy(nc2_angle_M_plot, res_path, 'nc2_activation_angle')
            save_list_as_npy(robust_nc2_angle_M_plot, res_path, 'robust_activation_nc2_angle')
            save_list_as_npy(nc2_angle_W_plot, res_path, 'nc2_classifier_angle')
            save_list_as_npy(robust_nc2_angle_W_plot, res_path, 'robust_classifier_nc2_angle')

            # save results
            save_list_as_npy(nc3_plot, res_path, 'nc3')
            save_list_as_npy(robust_nc3_plot, res_path, 'robust_nc3')

            # save results
            save_list_as_npy(nc4_plot, res_path, 'nc4')
            save_list_as_npy(robust_nc4_plot, res_path, 'robust_nc4')

        if args.neural_collapse_targeted:
            # save results 
            save_list_as_npy(targeted_nc1_plot, res_path, 'targeted_nc1')

            # save results
            save_list_as_npy(targeted_nc2_norm_M_plot, res_path, 'targeted_nc2_norm_M')
            save_list_as_npy(targeted_nc2_norm_W_plot, res_path, 'targeted_nc2_norm_W')

            # save results
            save_list_as_npy(targeted_nc2_angle_M_plot, res_path, 'targeted_nc2_angle_M')
            save_list_as_npy(targeted_nc2_angle_W_plot, res_path, 'targeted_nc2_angle_W')

            # save results
            save_list_as_npy(targeted_nc3_plot, res_path, 'targeted_nc3')

            # save results
            save_list_as_npy(targeted_nc4_plot, res_path, 'targeted_nc4')


            # save results 
            save_list_as_npy(clean_target_dist_plot, res_path, 'clean_target_dist')
            save_list_as_npy(clean_target_angle_plot, res_path, 'clean_target_angle')
            save_list_as_npy(clean_target_dist_plot_abs, res_path, 'clean_target_dist_abs')
            save_list_as_npy(clean_target_angle_plot_abs, res_path, 'clean_target_angle_abs')
            save_list_as_npy(target_muG_dist_plot, res_path, 'target_muG_dist')
            save_list_as_npy(target_muG_angle_plot, res_path, 'target_muG_angle')
            
        if args.neural_collapse_gaussian:
            # save results
            save_list_as_npy(gaussian_label_nc1_plot, res_path, 'new_gaussian_label_nc1')
            save_list_as_npy(gaussian_label_nc2_norm_M_plot, res_path, 'new_gaussian_label_nc2_activation_norm')
            save_list_as_npy(gaussian_label_nc2_norm_W_plot, res_path, 'new_gaussian_label_nc2_classifier_norm')
            save_list_as_npy(gaussian_label_nc2_angle_M_plot, res_path, 'new_gaussian_label_nc2_activation_angle')
            save_list_as_npy(gaussian_label_nc2_angle_W_plot, res_path, 'new_gaussian_label_nc2_classifier_angle')
            save_list_as_npy(gaussian_label_nc3_plot, res_path, 'new_gaussian_label_nc3')
            save_list_as_npy(gaussian_label_nc4_plot, res_path, 'new_gaussian_label_nc4')

            save_list_as_npy(linf_label_nc1_plot, res_path, 'linf_label_nc1')
            save_list_as_npy(linf_label_nc2_norm_M_plot, res_path, 'linf_label_nc2_activation_norm')
            save_list_as_npy(linf_label_nc2_norm_W_plot, res_path, 'linf_label_nc2_classifier_norm')
            save_list_as_npy(linf_label_nc2_angle_M_plot, res_path, 'linf_label_nc2_activation_angle')
            save_list_as_npy(linf_label_nc2_angle_W_plot, res_path, 'linf_label_nc2_classifier_angle')
            save_list_as_npy(linf_label_nc3_plot, res_path, 'linf_label_nc3')
            save_list_as_npy(linf_label_nc4_plot, res_path, 'linf_label_nc4')

if __name__ == '__main__':
    main()