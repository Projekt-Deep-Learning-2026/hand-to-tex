from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


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
    def get_onnx_export_configs(self) -> list[OnnxExportConfiguration]:
        raise NotImplementedError
