import pytest
import torch

from pl_bolts.models.vision import GPT2
from pl_bolts.models.vision.image_gpt.igpt_module import ImageGPT

# ImageGPT, SemSegment, UNet

seq_len = 17
batch_size = 32
vocab_size = 16
classify = True  # figure out a way to test with classify as false also
image_size = (1, 28, 28)

models = {
    "gpt2": {
        "model": GPT2(
            embed_dim=16, heads=2, layers=2, num_positions=17, vocab_size=16, num_classes=10, classify=classify
        ),
        "input": torch.randint(0, vocab_size, (seq_len, batch_size)),
        "output_dim": (17, 32, 16) if not classify else (32, 10),
    },
    "i-gpt": {
        "model": ImageGPT(),
        "input": torch.rand(batch_size, *image_size),
        "output_dim": (784, 32, 16),
    },
}


def _create_model(model_name):
    if model_name in models.keys():
        return models[model_name]["model"]
    else:
        raise ValueError("enter a valid model name")


def _get_input(model_name):
    """returns the expected input for the corresponding model."""
    if model_name in models.keys():
        return models[model_name]["input"]
    else:
        raise ValueError("enter a valid model name")


def _get_output_size(model_name):
    """returns the expected output size for the corresponding model."""
    if model_name in models.keys():
        return models[model_name]["output_dim"]
    else:
        raise ValueError("enter a valid model name")


model_to_test = ["gpt2", "i-gpt"]


@pytest.mark.parametrize("model_name", model_to_test)
def test_model_forward(catch_warnings, model_name: str):
    """
    Tests forward pass of a model
        1. Output should be desired shape
        2. Output should not have any NaN values
    """
    model = _create_model(model_name)
    model.eval()

    inputs = _get_input(model_name=model_name)
    output_size = _get_output_size(model_name=model_name)

    outputs = model(inputs)

    assert outputs.shape == torch.Size([*output_size])
    assert not torch.isnan(outputs).any(), "Output included NaNs"


@pytest.mark.parametrize("model_name", model_to_test)
def test_model_backward(catch_warnings, model_name: str):
    """
    Tests backward pass of a model
        1. none of the gradients should be NaN
        2. number of gradients should be the same as number of parameters
    """
    inputs = _get_input(model_name=model_name)

    model = _create_model(model_name)
    num_params = sum(x.numel() for x in model.parameters())
    model = model.train()

    outputs = model(inputs)

    # not sure what this is
    if isinstance(outputs, tuple):
        outputs = torch.cat(outputs)
    outputs.mean().backward()
    for n, x in model.named_parameters():
        assert x.grad is not None, f"No gradient for {n}"
    num_grad = sum(x.grad.numel() for x in model.parameters() if x.grad is not None)

    assert num_params == num_grad, "Some parameters are missing gradients"
