from enum import Enum
import torch

class _UNSET_T:
    def __repr__(self) -> str: return "UNSET"
UNSET = _UNSET_T()


class Mode(Enum):
    profile = 0
    execute = 1

class Schedule(Enum):
    W = 0
    S = 1
    
class ReduceType(Enum):
    Mean = 0
    Max = 1
    Min = 2
    L2Norm = 3
    Sum = 4
    
class ElementwiseBinaryOpType(Enum):
    Maximum = 0
    Minimum = 1
    Add = 2
    Mul = 3
    

class ElementwiseOpType(Enum):
    Relu = 0
    Sigmoid = 1
    Silu = 2
    Abs = 3
    Add_Mul = 4


class QuantizationType(Enum):
    BF16 = 0
    FP8_E5M2 = 1
    FP8_E4M3 = 2

INDENT = "    "
def indent_block(text: str, level: int = 1) -> str:
    """
    Indent a multi-line text block by the given indentation level.
    """
    prefix = INDENT * level
    lines = text.splitlines()
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def is_hopper_or_newer():
    """
    Check if the current CUDA device is Hopper architecture or newer.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Hopper architecture has compute capability 9.0 and above
    return (major > 9) or (major == 9 and minor >= 0)

def is_hopper():
    """
    Check if the current CUDA device is exactly Hopper architecture.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Hopper architecture has compute capability 9.0
    return (major == 9 and minor == 0)
