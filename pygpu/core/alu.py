"""
Arithmetic Logic Unit (ALU) - Handles all arithmetic operations with precision control.

The ALU enforces floating-point precision limits using NumPy scalar types,
preventing the use of Python's native 64-bit floats when simulating lower precision.
"""

from enum import Enum
from typing import Union
import numpy as np


class Precision(Enum):
    """Supported floating-point precision modes."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    
    @property
    def dtype(self) -> type:
        """Get the corresponding NumPy dtype."""
        return {
            Precision.FLOAT16: np.float16,
            Precision.FLOAT32: np.float32,
            Precision.FLOAT64: np.float64,
        }[self]
    
    @property
    def bits(self) -> int:
        """Get the number of bits for this precision."""
        return {
            Precision.FLOAT16: 16,
            Precision.FLOAT32: 32,
            Precision.FLOAT64: 64,
        }[self]


class ALU:
    """
    Arithmetic Logic Unit that performs operations with precision enforcement.
    
    All operations cast results to the specified precision, simulating
    the behavior of real GPU ALUs that have fixed precision modes.
    """
    
    def __init__(self, precision: Precision = Precision.FLOAT32):
        self.precision = precision
        self._dtype = precision.dtype
    
    def _cast(self, value: Union[int, float, np.number]) -> np.number:
        """Cast a value to the ALU's precision."""
        return self._dtype(value)
    
    def add(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Add two values."""
        return self._cast(self._cast(a) + self._cast(b))
    
    def sub(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Subtract b from a."""
        return self._cast(self._cast(a) - self._cast(b))
    
    def mul(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Multiply two values."""
        return self._cast(self._cast(a) * self._cast(b))
    
    def div(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Divide a by b (floating point)."""
        if b == 0:
            return self._cast(np.inf if a >= 0 else -np.inf)
        return self._cast(self._cast(a) / self._cast(b))
    
    def idiv(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Integer divide a by b (floor division)."""
        if b == 0:
            return self._cast(np.inf if a >= 0 else -np.inf)
        return self._cast(int(self._cast(a)) // int(self._cast(b)))
    
    def mod(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> np.number:
        """Modulo operation."""
        return self._cast(self._cast(a) % self._cast(b))
    
    def neg(self, a: Union[int, float, np.number]) -> np.number:
        """Negate a value."""
        return self._cast(-self._cast(a))
    
    def abs(self, a: Union[int, float, np.number]) -> np.number:
        """Absolute value."""
        return self._cast(np.abs(self._cast(a)))
    
    def sqrt(self, a: Union[int, float, np.number]) -> np.number:
        """Square root."""
        return self._cast(np.sqrt(self._cast(a)))
    
    def fma(self, a: Union[int, float, np.number], b: Union[int, float, np.number], 
            c: Union[int, float, np.number]) -> np.number:
        """Fused multiply-add: a * b + c (single rounding)."""
        # Note: True FMA would use a single rounding, but NumPy doesn't expose this
        return self._cast(self._cast(a) * self._cast(b) + self._cast(c))
    
    # Comparison operations (return boolean, used for branching)
    def eq(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test equality."""
        return self._cast(a) == self._cast(b)
    
    def ne(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test inequality."""
        return self._cast(a) != self._cast(b)
    
    def lt(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test less than."""
        return self._cast(a) < self._cast(b)
    
    def le(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test less than or equal."""
        return self._cast(a) <= self._cast(b)
    
    def gt(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test greater than."""
        return self._cast(a) > self._cast(b)
    
    def ge(self, a: Union[int, float, np.number], b: Union[int, float, np.number]) -> bool:
        """Test greater than or equal."""
        return self._cast(a) >= self._cast(b)
    
    # Bitwise operations (for integer types, but we simulate with floats)
    def band(self, a: int, b: int) -> int:
        """Bitwise AND."""
        return int(a) & int(b)
    
    def bor(self, a: int, b: int) -> int:
        """Bitwise OR."""
        return int(a) | int(b)
    
    def bxor(self, a: int, b: int) -> int:
        """Bitwise XOR."""
        return int(a) ^ int(b)
    
    def bnot(self, a: int) -> int:
        """Bitwise NOT."""
        return ~int(a)
    
    def shl(self, a: int, b: int) -> int:
        """Shift left."""
        return int(a) << int(b)
    
    def shr(self, a: int, b: int) -> int:
        """Shift right."""
        return int(a) >> int(b)
