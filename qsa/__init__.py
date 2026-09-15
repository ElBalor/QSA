from .complex_linear import ComplexLinear, spectral_clamp
from .attention import QSAttention, cp_superposition
from .rope import apply_phase_rope
from .cache import ComplexKVCache
from .layers import ComplexLayerNorm, ComplexFeedForward, QSDecoderBlock
from .heads import LogBornHead, ComplexClassifierHead
from .graft import ExpandedComplexLinear, zero_frozen_gradients
from .baseline import SoftmaxAttention, SoftmaxDecoderBlock
from .model import QSAReferenceLM, QSAConfig

__version__ = "0.2.0"
__author__ = "Heylel Yaka (Eric Heylel Danjuma Yaka)"
__description__ = (
    "Quantum Self-Attention: complex projective attention with constructive "
    "and destructive interference"
)

__all__ = [
    "ComplexLinear",
    "spectral_clamp",
    "QSAttention",
    "cp_superposition",
    "apply_phase_rope",
    "ComplexKVCache",
    "ComplexLayerNorm",
    "ComplexFeedForward",
    "QSDecoderBlock",
    "LogBornHead",
    "ComplexClassifierHead",
    "ExpandedComplexLinear",
    "zero_frozen_gradients",
    "SoftmaxAttention",
    "SoftmaxDecoderBlock",
    "QSAReferenceLM",
    "QSAConfig",
    "__version__",
]
