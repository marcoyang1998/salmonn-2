import torch
import torch.nn as nn

from .scaling import (
    ActivationDropoutAndLinear,
    SwooshLForward,
    SwooshRForward,
)


ZIPFORMER_BALANCED_PEFT_TARGET_MODULES = [
    "self_attn_weights.in_proj",
    "self_attn_weights.linear_pos",
    "self_attn1.in_proj",
    "self_attn1.out_proj",
    "self_attn2.in_proj",
    "self_attn2.out_proj",
    "feed_forward1.in_proj",
    "feed_forward1.out_proj.linear",
    "feed_forward2.in_proj",
    "feed_forward2.out_proj.linear",
    "feed_forward3.in_proj",
    "feed_forward3.out_proj.linear",
]


class PeftFriendlyActivationDropoutAndLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p=0.0,
        dropout_shared_dim: int = -1,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @classmethod
    def from_module(
        cls, module: ActivationDropoutAndLinear
    ) -> "PeftFriendlyActivationDropoutAndLinear":
        converted = cls(
            in_channels=module.weight.shape[1],
            out_channels=module.weight.shape[0],
            bias=module.bias is not None,
            activation=module.activation,
            dropout_p=module.dropout_p,
            dropout_shared_dim=module.dropout_shared_dim,
        )
        converted.linear = converted.linear.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            converted.linear.weight.copy_(module.weight)
            if module.bias is not None:
                converted.linear.bias.copy_(module.bias)
        return converted

    def forward(self, x: torch.Tensor):
        if self.activation == "SwooshL":
            x = SwooshLForward(x)
        elif self.activation == "SwooshR":
            x = SwooshRForward(x)
        else:
            assert False, self.activation
        return self.linear(x)


def convert_activation_dropout_and_linear_in_place(module: nn.Module) -> int:
    num_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, ActivationDropoutAndLinear):
            setattr(module, name, PeftFriendlyActivationDropoutAndLinear.from_module(child))
            num_replaced += 1
        else:
            num_replaced += convert_activation_dropout_and_linear_in_place(child)
    return num_replaced
