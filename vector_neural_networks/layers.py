import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.linalg import vector_norm

class VectorLayer(torch.nn.Module):
    def __init__(self, in_vector_size, out_vector_size, in_features, out_features):
        super().__init__()

        self.weight = Parameter(
            torch.Tensor(in_vector_size, out_vector_size, in_features, out_features)
        )
        self.weight.data.normal_(mean=0.0, std=1.0)
        # For every input there will be a transormation
        # matrix of size vector_size^2

    def forward(self, x: Tensor):
        """
        :param x: Input vector of size [batch x inputs_size X vector_size]
        :returns: Vector of size [batch X output_size X vector_size]
        """
        assemble = torch.einsum("biv,voif->bfo", x, self.weight)

        return assemble


class HalfSpaceProjection(torch.nn.Module):
    """
    Given a list of input vectors, project them onto the half space
    """

    def __init__(self, features: Tensor, vector_size: Tensor, epsilon: float = 1e-6):
        super().__init__()
        self.weight = Parameter(torch.Tensor(features, vector_size))
        self.weight.data.normal_(mean=0.0, std=1.0)
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """
        A bit different than the paper as they have an additional learned weight vector
        that transforms x. It's not clear to me why you would want that.
        """
        norm_squared = torch.clamp(
            torch.einsum("fv,fv->f", self.weight, self.weight), min=self.epsilon
        )
        projection = torch.einsum("bfv,fv->bf", x, self.weight) / norm_squared

        parallel_component = torch.einsum("bf,fv->bfv", projection, self.weight)
        projection = projection.unsqueeze(2)

        result = torch.where(projection < 0, x - parallel_component, x)
        return result


class VectorNormalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        v_norm =  vector_norm(x, ord=2, dim=2,keepdim=True)
        return x / v_norm
