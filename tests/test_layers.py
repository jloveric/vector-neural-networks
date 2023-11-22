import pytest
from vector_neural_networks.layers import VectorLayer
import torch

@pytest.mark.parametrize("in_vector_size", [1, 4])
@pytest.mark.parametrize("out_vector_size", [2, 3])
@pytest.mark.parametrize("in_features", [2, 3])
@pytest.mark.parametrize("out_features", [2, 5])
def test_vector_layer(in_vector_size, out_vector_size, in_features, out_features) :
    batch_size = 5
    layer = VectorLayer(in_vector_size=in_vector_size, out_vector_size=out_vector_size, in_features=in_features, out_features=out_features)
    in_vector = torch.rand(batch_size, in_features, in_vector_size)

    ans = layer(in_vector)

    assert ans.shape == torch.Size([batch_size, out_features, out_vector_size])

    print('ans', ans)

