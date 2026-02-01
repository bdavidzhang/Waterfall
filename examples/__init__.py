"""Example GPU kernels for PyGPU simulator.

Available examples:
- vector_add: Basic vector addition (GPU hello world)
- branch_divergence: Demonstrates SIMT branch divergence
- matrix_multiply: Simple matrix multiplication
- shared_memory: Shared memory usage for reduction
- precision_demo: Comparing float16/32/64 precision
- memory_patterns: Coalesced vs strided memory access
- python_kernels: Using the Python kernel compiler
- warp_scheduling: Visualizing warp scheduler behavior
"""

from examples.vector_add import run_vector_add
from examples.branch_divergence import run_divergence_demo
from examples.matrix_multiply import run_matrix_multiply
from examples.shared_memory import run_shared_memory_demo
from examples.precision_demo import run_precision_comparison
from examples.memory_patterns import run_memory_patterns_demo
from examples.python_kernels import run_python_kernel_demo
from examples.warp_scheduling import run_warp_scheduling_demo

__all__ = [
    'run_vector_add',
    'run_divergence_demo',
    'run_matrix_multiply',
    'run_shared_memory_demo',
    'run_precision_comparison',
    'run_memory_patterns_demo',
    'run_python_kernel_demo',
    'run_warp_scheduling_demo',
]
