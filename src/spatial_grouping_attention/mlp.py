# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py

# Modified by @rhoadesScholar, 2025-07-24

from typing import Callable, Optional

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP."""
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out + self.shortcut(x)
