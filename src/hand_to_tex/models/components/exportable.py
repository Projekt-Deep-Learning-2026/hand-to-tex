from abc import ABC

from hand_to_tex.models.components.base import BaseDecoderModel


class OnnxExportable(BaseDecoderModel, ABC):
    """
    Describes an interface for onnx-exporting script to make the nn.Module totally abstract
    from script logic
    """
