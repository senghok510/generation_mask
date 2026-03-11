from types import SimpleNamespace

import torch

from storyboard.features import _coerce_embedding_tensor


def test_coerce_embedding_tensor_accepts_tensor() -> None:
    tensor = torch.randn(2, 4)
    assert _coerce_embedding_tensor(tensor, output_name="image_embeds") is tensor


def test_coerce_embedding_tensor_accepts_named_attribute() -> None:
    tensor = torch.randn(2, 4)
    output = SimpleNamespace(image_embeds=tensor)
    assert _coerce_embedding_tensor(output, output_name="image_embeds") is tensor


def test_coerce_embedding_tensor_accepts_pooler_output() -> None:
    tensor = torch.randn(2, 4)
    output = SimpleNamespace(pooler_output=tensor)
    assert _coerce_embedding_tensor(output, output_name="image_embeds") is tensor
