import torch


def _create_model(model_name):
    return None


def _get_input_size(model_name):
    """returns the expected input size for the corresponding model."""
    pass


def _get_output_size(model_name):
    """returns the expected output size for the corresponding model."""
    pass


def test_model_forward(model_name: str, batch_size: int):
    """
    Tests forward pass of a model
        1. Output should be desired shape
        2. Output should not have any NaN values
    """
    model = _create_model(model_name)
    model.eval()

    input_size = _get_input_size(model_name=model_name)
    output_size = _get_output_size(model_name=model_name)
    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)

    assert outputs.shape == torch.Size([batch_size, *output_size])
    assert not torch.isnan(outputs).any(), "Output included NaNs"


def test_model_backward(model_name: str, batch_size: int):
    """
    Tests backward pass of a model
        1. none of the gradients should be NaN
        2. number of gradients should be the same as number of parameters
    """
    input_size = _get_input_size(model_name=model_name)

    model = _create_model(model_name, pretrained=False, num_classes=42)
    num_params = sum(x.numel() for x in model.parameters())
    model.train()

    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)

    # not sure what this is
    if isinstance(outputs, tuple):
        outputs = torch.cat(outputs)
    outputs.mean().backward()
    for n, x in model.named_parameters():
        assert x.grad is not None, f"No gradient for {n}"
    num_grad = sum(x.grad.numel() for x in model.parameters() if x.grad is not None)

    assert num_params == num_grad, "Some parameters are missing gradients"
