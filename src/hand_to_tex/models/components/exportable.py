from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class OnnxExportConfiguration:
    """Component's configuration describing script format for onnx-export"""

    name: str
    export_fun: Callable

    dummy_inputs: tuple
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]


class OnnxExportable(ABC):
    """
    Describes an interface for onnx-exporting script to make the nn.Module totally abstract
    from script logic
    """

    @abstractmethod
    def get_onnx_export_configs(self, device: str = "cpu") -> list[OnnxExportConfiguration]:
        pass


class DynamicOnnxExportWrapper(nn.Module):
    """Universal wrapper for `nn.Module`, allows us to point the onnx-export procedure
    to arbitrary function that replaces the `forward` method, useful, when dealing with
    models that perform non-standard export (like encoder-decoder separation)
    """

    def __init__(self, base: nn.Module, export_method: Callable) -> None:
        super().__init__()
        self.model = base
        self.method = export_method

    def forward(self, *args, **kwargs):
        return self.method(*args, **kwargs)
