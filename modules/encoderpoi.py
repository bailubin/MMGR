from torch_geometric.nn import Set2Set
import torch.nn as nn

class GCNSet2SetNoGCNNet(nn.Module):
    def __init__(self, projection_dim, dim_mlp=256):
        super(GCNSet2SetNoGCNNet, self).__init__()

        self.set2set = Set2Set(64, processing_steps=5)
        self.linear1 = nn.Linear(128, 256, bias=True)
        self.projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, projection_dim, bias=True),
        )

    def forward(self, data):
        out = self.set2set(data.x, data.batch.long()).float()
        out = self.linear1(out)
        out = self.projector(out)
        return out
