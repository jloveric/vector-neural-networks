import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.parameter import Parameter

class VectorLayer(torch.nn.Module) :
    def __init__(self, in_vector_size, out_vector_size, in_features, out_features) :
        super().__init__()
        
        self.weight = Parameter(torch.Tensor(in_vector_size, out_vector_size, in_features, out_features))
        # For every input there will be a transormation
        # matrix of size vector_size^2

        input_size = in_vector_size*out_vector_size*in_features


    def forward(self, x: Tensor):
        """
        Input vector of size [basis x inputs_size X vector_size]
        """
        assemble = torch.einsum("biv,voif->bfo", x, self.weight)