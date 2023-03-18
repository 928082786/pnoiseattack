import torch
from torchvision import transforms

import os
import torch.nn as nn
import copy
import torch.nn.functional as F
import utils
import resnet
import numpy as np
from noises import perlin
from utils import progress_bar


def add_perlin(x, a):
    pdata = x.copy()
    shape = pdata.shape[:-1]

    pnoise = perlin.perlin_noise(shape=shape)
    con_pnoise = np.array([pnoise] * pdata.shape[-1]).transpose(1, 2, 0)
    p_d = np.clip(pdata + con_pnoise * a, 0, 1)
    pdata = p_d
    return pdata

def eval(netC, testloader, device):
    acc = 0
    total_sample = 0
    total_correct = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds = netC(inputs)
        correct = torch.sum(torch.argmax(preds, 1) == targets)
        total_correct += correct
        acc = total_correct * 100.0 / total_sample

    return acc


def main(mode='cifar', batch_size=128):
    # Prepare arguments
    if mode == 'cifar':
        config_path = './Configs/cifar10.yaml'
        config = utils.get_config(config_path)
        device = config['device']
        cifar_path = config['datapath']
        target_class = config['target_class']
        noise_a = config['noise_alpha']

        num_classes = 10

        (x_train, y_train), (x_test, y_test), min_, max_ = utils.load_cifar10(cifar_path)
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # prepare data
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        testset = utils.OwnDataset(x_test, y_test, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

        not_target_test = np.arange(x_test.shape[0])[y_test != target_class]
        poison_ind_test = np.random.permutation(not_target_test)
        p_x_test = x_test.copy()
        p_y_test = y_test.copy()
        for i in range(poison_ind_test.shape[0]):
            p_x_test[poison_ind_test[i]] = add_perlin(x_test[poison_ind_test[i]], a=0.5)
            p_y_test[poison_ind_test[i]] = target_class

        p_testset = utils.OwnDataset(p_x_test, p_y_test, transform=transform_test)
        p_testloader = torch.utils.data.DataLoader(p_testset, batch_size=batch_size, shuffle=False, num_workers=0)

        # load model
        netC = resnet.ResNet18().to(device)

        if os.path.exists('./checkpoint/{}.pth'.format(config['dataset'])):
            checkpoint = torch.load('./checkpoint/{}.pth'.format(config['dataset']))
            netC.load_state_dict(checkpoint['net'])

        netC.eval()
        netC.requires_grad_(False)
    else:
        raise NotADirectoryError


    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(testloader):
        inputs = inputs.to(device)
        netC(inputs)
        progress_bar(batch_idx, len(testloader))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    outfile = "./Log/{}_results.txt".format(config['dataset'])
    with open(outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(device)
            clean_acc = eval(net_pruned, testloader, device)
            bd_acc = eval(net_pruned, p_testloader, device)
            # clean, bd = eval(net_pruned, identity_grid, noise_grid, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean_acc, bd_acc))

if __name__ == "__main__":
    main()
