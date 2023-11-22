import pytest
from vector_neural_networks.layers import VectorLayer
import torch

def test_vector_layer() :

    layer = VectorLayer(in_vector_size=3, out_vector_size=3, in_features=10, out_features=4)
    in_vector = torch.rand(5, 10, 3)

    ans = layer(in_vector)

    assert ans.shape == torch.Size([5, 4, 3])

    print('ans', ans)

