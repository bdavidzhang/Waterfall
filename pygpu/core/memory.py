"""
GPU Memory Model.

Implements the memory hierarchy:
- GlobalMemory: Large, slow memory accessible by all threads
- SharedMemory: Small, fast memory shared within a block
"""

from typing import Optional, List, Set
import numpy as np

from pygpu.core.alu import Precision


class GlobalMemory:
    """
    Global memory - accessible by all threads across all blocks.
    
    This is the main GPU memory (VRAM), which is large but has high latency.
    In real GPUs, global memory access can take hundreds of cycles.
    """
    
    DEFAULT_SIZE = 65536  # 64KB default
    LATENCY_CYCLES = 100  # Simulated memory latency
    
    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        precision: Precision = Precision.FLOAT32,
    ):
        self.size = size
        self.precision = precision
        self._dtype = precision.dtype
        self._data = np.zeros(size, dtype=self._dtype)
        
        # Access tracking for visualization
        self._read_addresses: Set[int] = set()
        self._write_addresses: Set[int] = set()
        self._access_history: List[tuple] = []  # (cycle, addr, 'r'/'w')
    
    def read(self, address: int) -> np.number:
        """Read a value from memory."""
        if 0 <= address < self.size:
            self._read_addresses.add(address)
            return self._data[address]
        raise IndexError(f"Memory address {address} out of range [0, {self.size})")
    
    def write(self, address: int, value: float):
        """Write a value to memory."""
        if 0 <= address < self.size:
            self._write_addresses.add(address)
            self._data[address] = self._dtype(value)
        else:
            raise IndexError(f"Memory address {address} out of range [0, {self.size})")
    
    def load_array(self, data: List[float], start_address: int = 0):
        """Load an array of values into memory starting at the given address."""
        for i, value in enumerate(data):
            if start_address + i < self.size:
                self._data[start_address + i] = self._dtype(value)
    
    def read_array(self, start_address: int, length: int) -> np.ndarray:
        """Read an array of values from memory."""
        end = min(start_address + length, self.size)
        return self._data[start_address:end].copy()
    
    def clear_access_tracking(self):
        """Clear the access tracking sets (called each cycle)."""
        self._read_addresses.clear()
        self._write_addresses.clear()
    
    def record_cycle(self, cycle: int):
        """Record access history for this cycle."""
        for addr in self._read_addresses:
            self._access_history.append((cycle, addr, 'r'))
        for addr in self._write_addresses:
            self._access_history.append((cycle, addr, 'w'))
    
    def get_heatmap(self, start: int = 0, end: Optional[int] = None) -> List[int]:
        """
        Get access frequency for a memory range.
        
        Returns a list of access counts for visualization.
        """
        end = end or self.size
        counts = [0] * (end - start)
        for _, addr, _ in self._access_history:
            if start <= addr < end:
                counts[addr - start] += 1
        return counts
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        non_zero = np.count_nonzero(self._data)
        return f"GlobalMemory(size={self.size}, non_zero={non_zero})"


class SharedMemory:
    """
    Shared memory - fast memory accessible by threads within the same block.
    
    Shared memory is much faster than global memory (typically 1-2 cycles)
    but is limited in size and only visible to threads in the same block.
    """
    
    DEFAULT_SIZE = 4096  # 4KB default per block
    LATENCY_CYCLES = 1   # Fast access
    
    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        precision: Precision = Precision.FLOAT32,
    ):
        self.size = size
        self.precision = precision
        self._dtype = precision.dtype
        self._data = np.zeros(size, dtype=self._dtype)
        
        # Access tracking
        self._read_addresses: Set[int] = set()
        self._write_addresses: Set[int] = set()
    
    def read(self, address: int) -> np.number:
        """Read a value from shared memory."""
        if 0 <= address < self.size:
            self._read_addresses.add(address)
            return self._data[address]
        raise IndexError(f"Shared memory address {address} out of range [0, {self.size})")
    
    def write(self, address: int, value: float):
        """Write a value to shared memory."""
        if 0 <= address < self.size:
            self._write_addresses.add(address)
            self._data[address] = self._dtype(value)
        else:
            raise IndexError(f"Shared memory address {address} out of range [0, {self.size})")
    
    def clear(self):
        """Clear all shared memory (called when block finishes)."""
        self._data.fill(0)
        self._read_addresses.clear()
        self._write_addresses.clear()
    
    def clear_access_tracking(self):
        """Clear access tracking for this cycle."""
        self._read_addresses.clear()
        self._write_addresses.clear()
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        non_zero = np.count_nonzero(self._data)
        return f"SharedMemory(size={self.size}, non_zero={non_zero})"
