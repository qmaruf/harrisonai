import sys

sys.path.append("src")
import pytest
import torch
from network import ChannelReduce, ConvRelu, PetNet


def test_conv_relu():
    conv_relu = ConvRelu(in_channels=3, out_channels=10, kernel=3, padding=1)
    x = torch.randn(1, 3, 64, 64)
    output = conv_relu(x)
    assert output.shape == torch.Size([1, 10, 64, 64])


def test_channel_reduce():
    channel_reduce = ChannelReduce(in_channels=128, out_channels=1)
    x = torch.randn(1, 128, 32, 32)
    output = channel_reduce(x)
    assert output.shape == torch.Size([1, 1, 32, 32])


def test_petnet_output():
    pet_net = PetNet()
    x = torch.randn(1, 3, 256, 256)
    seg_output, clf_output = pet_net(x)
    assert seg_output.shape == torch.Size([1, 1, 256, 256])
    assert clf_output.shape == torch.Size([1, 39])


if __name__ == "__main__":
    pytest.main()
