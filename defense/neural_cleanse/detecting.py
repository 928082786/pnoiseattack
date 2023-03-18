import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms as transforms
import utils

import resnet
import os
import numpy as np
from utils import progress_bar


class RegressionModel(nn.Module):
    def __init__(self, config):
        self._EPSILON = config['EPSILON']
        super(RegressionModel, self).__init__()

        self.classifier = self._get_classifier(config)
        self.normalizer = self._get_normalize(config)
        self.denormalizer = self._get_denormalize(config)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, config):
        if config['dataset'] == "cifar10":
            classifier = resnet.ResNet18()
        elif config['dataset'] == "gtsrb":
            classifier = resnet.ResNet18(num_classes=43)
        else:
            raise Exception("Invalid Dataset")
        # Load pretrained classifie
        checkpoint = torch.load('../../checkpoint/{}.pth'.format(config['dataset']))
        classifier.load_state_dict(checkpoint['net'])

        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(config['device'])

    def _get_denormalize(self, config):
        if config['dataset'] == "cifar10":
            denormalizer = utils.Denormalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif config['dataset'] == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, config):
        if config['dataset'] == "cifar10":
            normalizer = utils.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif config['dataset'] == "gtsrb" :
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, config):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = config['init_cost']
        self.cost_multiplier_up = config['cost_multiplier']
        self.cost_multiplier_down = config['cost_multiplier'] ** 1.5

    def reset_state(self, config):
        self.cost = config['init_cost']
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))


def train(mode, config, batch_size):
    if mode == 'cifar':
        device = config['device']
        cifar_path = config['datapath']
        (_, _), (x_test, y_test), min_, max_ = utils.load_cifar10(cifar_path)
        y_test = np.argmax(y_test, axis=1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise NotImplemented

    testset = utils.OwnDataset(x_test, y_test, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build regression model
    regression_model = RegressionModel(config).to(device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=1e-3, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(config)

    for epoch in range(100):
        early_stop = train_step(regression_model, optimizerR, testloader, recorder, epoch, config)
        if early_stop:
            break

    return recorder, config


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, config):
    print("Epoch {} - Label: {} | {}:".format(epoch, config['target_class'], config['dataset']))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(config['device'])
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(config['device']) * config['target_class']
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), 1)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
        progress_bar(batch_idx, len(dataloader))

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= config['atk_succ_threshold'] and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        print(" Updated !!!")

    # Show information
    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
        )
    )

    # Check early stop
    if config['early_stop']:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= config['early_stop_threshold'] * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= config['early_stop_patience']
        ):
            print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= config['atk_succ_threshold']:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= config['patience']:
                recorder.reset_state(config)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= config['atk_succ_threshold']:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= config['patience']:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= config['patience']:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag

if __name__ == "__main__":
    pass
