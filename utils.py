from torch import optim

from net import NNet

def create_nn_array(sizes):
    neural_nets = [NNet(sizes[i], num_of_classes=3) for i in range(len(sizes))]
    optimizers = [optim.Adam(net.parameters()) for net in neural_nets]
    return neural_nets, optimizers
def get_sizes(population):
    return [x.count(1) for x in population]




def get_probab(losses):
    losses = [x / sum(losses) for x in losses]
    return losses

