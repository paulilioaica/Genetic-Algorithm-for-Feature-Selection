from torch import optim
from genetic import device
from net import NNet


def create_nn_array(sizes):
    neural_nets = [NNet(sizes[i], num_of_classes=3).to(device) for i in range(len(sizes))]
    optimizers = [optim.Adam(net.parameters(), lr=0.01) for net in neural_nets]
    return neural_nets, optimizers


def get_sizes(population):
    return [x.count(1) for x in population]


def get_probab(losses):
    losses = [x / sum(losses) for x in losses]
    return losses
