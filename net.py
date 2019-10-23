import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):
    def __init__(self, input_dim, num_of_classes):
        super().__init__()
        self.mid_layer = nn.Linear(input_dim, 3* input_dim)
        self.mid_layer_1 = nn.Linear(3*input_dim, 3*input_dim)
        self.out_layer = nn.Linear(3*input_dim, num_of_classes)

    def forward(self, x):
        x = self.mid_layer(x)
        x = F.sigmoid(x)
        x = self.mid_layer_1(x)
        x = F.sigmoid(x)
        x = self.out_layer(x)
        return F.softmax(x, dim=1)



