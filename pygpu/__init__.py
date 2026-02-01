"""
PyGPU - A cycle-accurate GPU simulator for educational purposes.

This simulator implements the SIMT (Single Instruction, Multiple Threads) 
execution model used by modern GPUs like NVIDIA CUDA.
"""

from pygpu.gpu import GPU
from pygpu.core.thread import Thread, ThreadStatus
from pygpu.core.warp import Warp
from pygpu.core.block import Block
from pygpu.core.sm import StreamingMultiprocessor
from pygpu.core.memory import GlobalMemory, SharedMemory
from pygpu.core.instruction import Instruction, OpCode

__version__ = "0.1.0"
__all__ = [
    "GPU",
    "Thread",
    "ThreadStatus",
    "Warp",
    "Block",
    "StreamingMultiprocessor",
    "GlobalMemory",
    "SharedMemory",
    "Instruction",
    "OpCode",
]
