"""
Precision Comparison Example

This example demonstrates the difference between float16, float32, and float64
precision modes. It shows how numerical precision affects computation results,
especially for operations that accumulate errors.
"""

from rich.console import Console
from rich.table import Table
import numpy as np

from pygpu import GPU
from pygpu.core.instruction import Program

console = Console()


def create_precision_test_program() -> Program:
    """
    Create a program that performs operations sensitive to precision.
    
    Computes: result = ((x + small) - x) * scale
    
    With perfect precision, this equals small * scale.
    With limited precision, the small value may be lost when added to x.
    """
    assembly = """
        THREAD_ID R0
        
        # Each thread tests with different base values
        # R1 = thread_id * 1000 (large base value)
        MOV R1, 1000
        MUL R1, R0, R1
        
        # R2 = 0.001 (small value that may be lost)
        MOV R2, 0.001
        
        # R3 = R1 + R2 (add small to large)
        ADD R3, R1, R2
        
        # R4 = R3 - R1 (subtract large, should recover small)
        SUB R4, R3, R1
        
        # R5 = R4 * 1000000 (scale up to make error visible)
        MOV R5, 1000000
        MUL R6, R4, R5
        
        # Store result
        STORE R6, [R0]
        
        RET
    """
    return Program.from_assembly(assembly)


def run_precision_test(precision: str, num_threads: int = 8) -> list:
    """Run precision test with specified precision mode."""
    gpu = GPU(precision=precision)
    
    program = create_precision_test_program()
    gpu.load_program(program)
    
    gpu.launch(blocks=1, threads_per_block=num_threads, threads_per_warp=num_threads)
    gpu.run()
    
    return [gpu.global_memory.read(i) for i in range(num_threads)]


def run_precision_comparison():
    """Run the precision comparison demonstration."""
    console.print("[bold cyan]Precision Comparison Example[/bold cyan]")
    console.print("=" * 70)
    console.print()
    console.print("This example shows how floating-point precision affects computations.")
    console.print()
    console.print("[yellow]Operation:[/yellow] result = ((base + small) - base) × scale")
    console.print("  - base  = thread_id × 1000")
    console.print("  - small = 0.001")
    console.print("  - scale = 1,000,000")
    console.print()
    console.print("[yellow]Expected result:[/yellow] 0.001 × 1,000,000 = 1000 for all threads")
    console.print()
    
    # Test each precision mode
    precisions = ['float16', 'float32', 'float64']
    results = {}
    
    for prec in precisions:
        console.print(f"Testing {prec}...", end=" ")
        results[prec] = run_precision_test(prec)
        console.print("[green]done[/green]")
    
    console.print()
    
    # Create results table
    table = Table(title="Results by Precision Mode")
    table.add_column("Thread", justify="center")
    table.add_column("Base Value", justify="right")
    table.add_column("float16", justify="right")
    table.add_column("float32", justify="right")
    table.add_column("float64", justify="right")
    table.add_column("Expected", justify="right")
    
    for i in range(8):
        base = i * 1000
        expected = 1000.0
        
        f16 = results['float16'][i]
        f32 = results['float32'][i]
        f64 = results['float64'][i]
        
        # Color code based on accuracy
        def format_val(val, expected):
            if abs(float(val) - expected) < 0.01:
                return f"[green]{float(val):.1f}[/green]"
            elif abs(float(val) - expected) < 100:
                return f"[yellow]{float(val):.1f}[/yellow]"
            else:
                return f"[red]{float(val):.1f}[/red]"
        
        table.add_row(
            str(i),
            str(base),
            format_val(f16, expected),
            format_val(f32, expected),
            format_val(f64, expected),
            str(expected),
        )
    
    console.print(table)
    console.print()
    
    # Explanation
    console.print("[bold]Analysis:[/bold]")
    console.print()
    console.print("  [green]float64[/green]: Has enough precision to preserve the small value")
    console.print("  [yellow]float32[/yellow]: May show minor errors for larger base values")
    console.print("  [red]float16[/red]: Only ~3 decimal digits of precision, significant errors")
    console.print()
    console.print("This demonstrates why GPU applications choose precision carefully:")
    console.print("  - [bold]float16[/bold]: Fastest, good for ML inference where accuracy is less critical")
    console.print("  - [bold]float32[/bold]: Default for most GPU computing")
    console.print("  - [bold]float64[/bold]: Scientific computing requiring high precision")


def run_accumulation_error_demo():
    """Demonstrate accumulation error across precision modes."""
    console.print("\n[bold cyan]Accumulation Error Demo[/bold cyan]")
    console.print("=" * 70)
    console.print()
    console.print("Summing 0.1 one thousand times (should equal 100.0):")
    console.print()
    
    for prec in ['float16', 'float32', 'float64']:
        dtype = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}[prec]
        
        # Simulate accumulation
        total = dtype(0.0)
        for _ in range(1000):
            total = dtype(total + dtype(0.1))
        
        error = abs(float(total) - 100.0)
        
        if error < 0.001:
            color = "green"
        elif error < 1.0:
            color = "yellow"
        else:
            color = "red"
        
        console.print(f"  {prec:8s}: {float(total):12.6f}  (error: [{color}]{error:.6f}[/{color}])")


if __name__ == "__main__":
    run_precision_comparison()
    run_accumulation_error_demo()
