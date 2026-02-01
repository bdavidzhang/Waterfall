"""
Python Kernel Compilation Example

This example demonstrates the KernelCompiler, which allows you to
write GPU kernels in Python syntax instead of assembly.

The compiler translates Python code to our GPU's instruction set.
"""

from rich.console import Console
from rich.syntax import Syntax
import inspect

from pygpu import GPU

console = Console()


# Simple kernels written in Python
def vector_scale_kernel(thread_id, global_mem):
    """Scale each element by 2."""
    val = global_mem[thread_id]
    result = val * 2
    global_mem[thread_id + 100] = result


def conditional_kernel(thread_id, global_mem):
    """Conditional processing based on thread ID."""
    if thread_id < 4:
        result = thread_id * 10
    else:
        result = thread_id * 5
    global_mem[thread_id] = result


def arithmetic_kernel(thread_id, global_mem):
    """Demonstrate various arithmetic operations."""
    a = global_mem[thread_id]
    b = global_mem[thread_id + 10]
    
    # Various operations
    sum_val = a + b
    diff = a - b
    prod = a * b
    
    # Compound expression
    result = sum_val + prod - diff
    
    global_mem[thread_id + 100] = result


def loop_kernel(thread_id, global_mem):
    """Demonstrate loop compilation (for loop)."""
    total = 0
    for i in range(4):
        total = total + global_mem[thread_id + i * 8]
    global_mem[thread_id + 100] = total


def run_kernel_demo(kernel_func, name: str, input_data: list = None, num_threads: int = 8):
    """Run a kernel and show results."""
    console.print(f"\n[bold cyan]{name}[/bold cyan]")
    console.print("-" * 50)
    
    # Show the kernel source code
    source = inspect.getsource(kernel_func)
    # Remove the first line (def) indentation issues
    lines = source.split('\n')
    # Find minimum indentation
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    cleaned = '\n'.join(line[min_indent:] if len(line) > min_indent else line for line in lines)
    
    console.print("[yellow]Kernel source:[/yellow]")
    syntax = Syntax(cleaned, "python", theme="monokai", line_numbers=True)
    console.print(syntax)
    console.print()
    
    # Create GPU and compile kernel
    gpu = GPU(precision='float32')
    
    try:
        gpu.load_program(kernel_func)
    except Exception as e:
        console.print(f"[red]Compilation failed: {e}[/red]")
        return
    
    # Show compiled assembly
    console.print("[yellow]Compiled assembly:[/yellow]")
    console.print(f"[dim]{gpu.program}[/dim]")
    console.print()
    
    # Initialize input data if provided
    if input_data:
        gpu.global_memory.load_array(input_data, 0)
        console.print(f"[yellow]Input data:[/yellow] {input_data[:20]}...")
    
    # Launch kernel
    gpu.launch(blocks=1, threads_per_block=num_threads, threads_per_warp=num_threads)
    
    # Run
    cycles = gpu.run()
    console.print(f"[green]Completed in {cycles} cycles[/green]")
    
    # Show output
    if input_data:
        output = gpu.global_memory.read_array(100, num_threads)
        console.print(f"[yellow]Output:[/yellow] {[float(x) for x in output]}")
    else:
        output = gpu.global_memory.read_array(0, num_threads)
        console.print(f"[yellow]Output:[/yellow] {[float(x) for x in output]}")


def run_python_kernel_demo():
    """Run the Python kernel compilation demonstration."""
    console.print("[bold cyan]Python Kernel Compilation Example[/bold cyan]")
    console.print("=" * 70)
    console.print()
    console.print("The KernelCompiler translates Python kernels to GPU assembly.")
    console.print("This allows writing GPU code in familiar Python syntax!")
    console.print()
    
    # Demo 1: Vector scaling
    run_kernel_demo(
        vector_scale_kernel,
        "1. Vector Scaling",
        input_data=[float(i) for i in range(8)],
        num_threads=8
    )
    
    # Demo 2: Conditional
    run_kernel_demo(
        conditional_kernel,
        "2. Conditional Processing",
        num_threads=8
    )
    
    # Demo 3: Arithmetic
    input_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    input_b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    combined = input_a + [0.0] * 2 + input_b  # a at 0, b at 10
    run_kernel_demo(
        arithmetic_kernel,
        "3. Arithmetic Operations",
        input_data=combined,
        num_threads=8
    )
    
    # Demo 4: Loop
    # Prepare data: each thread will sum 4 values
    loop_data = [float(i) for i in range(64)]
    run_kernel_demo(
        loop_kernel,
        "4. Loop (for i in range)",
        input_data=loop_data,
        num_threads=8
    )
    
    console.print()
    console.print("[bold]Supported Python Features:[/bold]")
    console.print("  ✓ Variable assignments")
    console.print("  ✓ Arithmetic: +, -, *, /, %")
    console.print("  ✓ Memory access: global_mem[idx]")
    console.print("  ✓ Conditionals: if/else")
    console.print("  ✓ Loops: for i in range(...)")
    console.print("  ✓ Comparisons: ==, !=, <, <=, >, >=")
    console.print()
    console.print("[bold]Not Yet Supported:[/bold]")
    console.print("  ✗ Function calls")
    console.print("  ✗ Nested loops (coming soon)")
    console.print("  ✗ Shared memory (requires syntax extension)")


if __name__ == "__main__":
    run_python_kernel_demo()
