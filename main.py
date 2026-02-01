#!/usr/bin/env python3
"""
PyGPU - A cycle-accurate GPU simulator for educational purposes.

This main script provides an entry point to run examples or launch
the interactive visualization dashboard.

Usage:
    python main.py                    # Run all examples
    python main.py vector_add         # Run vector addition example
    python main.py interactive        # Run step-by-step vector addition
    python main.py divergence         # Run branch divergence demo
    python main.py matrix             # Run matrix multiplication
    python main.py shared             # Run shared memory demo
    python main.py precision          # Run precision comparison
    python main.py memory             # Run memory patterns demo
    python main.py kernels            # Run Python kernel compilation demo
    python main.py scheduling         # Run warp scheduling demo
    python main.py dashboard          # Launch interactive dashboard
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_banner():
    """Print the PyGPU banner."""
    banner_text = Text()
    banner_text.append("██████╗ ██╗   ██╗ ██████╗ ██████╗ ██╗   ██╗\n", style="bold cyan")
    banner_text.append("██╔══██╗╚██╗ ██╔╝██╔════╝ ██╔══██╗██║   ██║\n", style="bold cyan")
    banner_text.append("██████╔╝ ╚████╔╝ ██║  ███╗██████╔╝██║   ██║\n", style="bold cyan")
    banner_text.append("██╔═══╝   ╚██╔╝  ██║   ██║██╔═══╝ ██║   ██║\n", style="bold cyan")
    banner_text.append("██║        ██║   ╚██████╔╝██║     ╚██████╔╝\n", style="bold cyan")
    banner_text.append("╚═╝        ╚═╝    ╚═════╝ ╚═╝      ╚═════╝\n\n", style="bold cyan")
    banner_text.append("Cycle-Accurate GPU Simulator\n", style="bold white")
    banner_text.append("SIMT Execution | Branch Divergence | Memory Hierarchy", style="dim")
    
    console.print(Panel(banner_text, border_style="cyan"))


def run_vector_add_example():
    """Run the vector addition example."""
    from examples.vector_add import run_vector_add
    run_vector_add()


def run_vector_add_interactive_example():
    """Run the step-by-step vector addition example."""
    from examples.vector_add import run_vector_add_interactive
    run_vector_add_interactive()


def run_divergence_example():
    """Run the branch divergence example."""
    from examples.branch_divergence import run_divergence_demo
    run_divergence_demo()


def run_matrix_example():
    """Run the matrix multiplication example."""
    from examples.matrix_multiply import run_matrix_multiply
    run_matrix_multiply()


def run_shared_memory_example():
    """Run the shared memory example."""
    from examples.shared_memory import run_shared_memory_demo
    run_shared_memory_demo()


def run_precision_example():
    """Run the precision comparison example."""
    from examples.precision_demo import run_precision_comparison, run_accumulation_error_demo
    run_precision_comparison()
    run_accumulation_error_demo()


def run_memory_patterns_example():
    """Run the memory access patterns example."""
    from examples.memory_patterns import run_memory_patterns_demo
    run_memory_patterns_demo()


def run_python_kernels_example():
    """Run the Python kernel compilation example."""
    from examples.python_kernels import run_python_kernel_demo
    run_python_kernel_demo()


def run_warp_scheduling_example():
    """Run the warp scheduling example."""
    from examples.warp_scheduling import run_warp_scheduling_demo, run_multi_sm_demo
    run_warp_scheduling_demo()
    run_multi_sm_demo()


def run_dashboard():
    """Launch the interactive visualization dashboard."""
    from pygpu import GPU
    from pygpu.ui.dashboard import run_dashboard
    from examples.vector_add import create_vector_add_program
    
    print("Launching interactive GPU dashboard...")
    print("Use Space to step, R to run, P to pause, Q to quit")
    print()
    
    # Set up a simple simulation
    gpu = GPU(precision='float32')
    program = create_vector_add_program()
    gpu.load_program(program)
    
    # Initialize data
    gpu.global_memory.load_array(list(range(16)), start_address=0)
    gpu.global_memory.load_array([100 + i for i in range(16)], start_address=100)
    
    # Launch
    gpu.launch(blocks=2, threads_per_block=8)
    
    # Run dashboard
    run_dashboard(gpu)


def run_all_examples():
    """Run all examples in sequence."""
    console.print("\n" + "=" * 60)
    console.print("[bold magenta]EXAMPLE 1: Vector Addition (Step-by-Step)[/bold magenta]")
    console.print("=" * 60 + "\n")
    run_vector_add_interactive_example()
    
    console.print("\n\n" + "=" * 60)
    console.print("[bold magenta]EXAMPLE 2: Branch Divergence[/bold magenta]")
    console.print("=" * 60 + "\n")
    run_divergence_example()
    
    console.print("\n\n" + "=" * 60)
    console.print("[bold magenta]EXAMPLE 3: Matrix Multiplication[/bold magenta]")
    console.print("=" * 60 + "\n")
    run_matrix_example()


def print_help():
    """Print usage information."""
    console.print(__doc__)
    console.print("\n[bold]Available commands:[/bold]")
    console.print("  [cyan]all[/cyan]         - Run all examples")
    console.print("  [cyan]vector_add[/cyan]  - Vector addition (C = A + B)")
    console.print("  [cyan]interactive[/cyan] - Step-by-step vector addition")
    console.print("  [cyan]divergence[/cyan]  - Branch divergence demonstration")
    console.print("  [cyan]matrix[/cyan]      - Matrix multiplication")
    console.print("  [cyan]shared[/cyan]      - Shared memory usage demo")
    console.print("  [cyan]precision[/cyan]   - Precision comparison (f16/f32/f64)")
    console.print("  [cyan]memory[/cyan]      - Memory access patterns demo")
    console.print("  [cyan]kernels[/cyan]     - Python kernel compilation demo")
    console.print("  [cyan]scheduling[/cyan]  - Warp scheduling visualization")
    console.print("  [cyan]dashboard[/cyan]   - Interactive visualization (requires Textual)")
    console.print("  [cyan]help[/cyan]        - Show this help message")


def main():
    """Main entry point."""
    print_banner()
    
    if len(sys.argv) < 2:
        # No arguments - run all examples
        run_all_examples()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        "all": run_all_examples,
        "vector_add": run_vector_add_example,
        "vector": run_vector_add_example,
        "interactive": run_vector_add_interactive_example,
        "step": run_vector_add_interactive_example,
        "divergence": run_divergence_example,
        "branch": run_divergence_example,
        "matrix": run_matrix_example,
        "matmul": run_matrix_example,
        "shared": run_shared_memory_example,
        "shared_memory": run_shared_memory_example,
        "precision": run_precision_example,
        "float": run_precision_example,
        "memory": run_memory_patterns_example,
        "memory_patterns": run_memory_patterns_example,
        "patterns": run_memory_patterns_example,
        "kernels": run_python_kernels_example,
        "python_kernels": run_python_kernels_example,
        "compile": run_python_kernels_example,
        "scheduling": run_warp_scheduling_example,
        "warp": run_warp_scheduling_example,
        "schedule": run_warp_scheduling_example,
        "dashboard": run_dashboard,
        "ui": run_dashboard,
        "help": print_help,
        "-h": print_help,
        "--help": print_help,
    }
    
    if command in commands:
        commands[command]()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
