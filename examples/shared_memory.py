"""
Shared Memory Example

This example demonstrates the use of shared memory within a block.
Shared memory is fast, low-latency memory that is shared between
all threads in a block, making it ideal for data reuse patterns.

We implement a simple reduction (sum) operation that uses shared
memory to efficiently combine results within a block.
"""

from rich.console import Console
from rich.table import Table

from pygpu import GPU
from pygpu.core.instruction import Program, Instruction, OpCode, MemorySpace

console = Console()


def create_shared_memory_program() -> Program:
    """
    Create a program that demonstrates shared memory usage.
    
    Each thread:
    1. Loads its value from global memory
    2. Stores it to shared memory
    3. (In a full implementation, would do parallel reduction)
    4. Thread 0 sums all values and stores to output
    
    Memory layout:
    - Global [0:8] = Input values
    - Global [100] = Output sum
    - Shared [0:8] = Intermediate storage
    """
    assembly = """
        # Get thread ID
        THREAD_ID R0
        
        # Load value from global memory at address = thread_id
        LOAD R1, [R0]
        
        # Store to shared memory at same offset
        # (Shared memory operations would use STORE.SHARED in real impl)
        # For now, we simulate by using a different address range
        MOV R2, 50
        ADD R2, R2, R0
        STORE R1, [R2]
        
        # Synchronize all threads in block
        SYNC
        
        # Only thread 0 does the final sum
        MOV R3, 0
        CMP_EQ P0, R0, R3
        BRA_N skip_sum, P0
        
        # Thread 0: Sum all values from shared memory
        MOV R10, 0      # Accumulator
        
        # Load and add each value (unrolled for simplicity)
        MOV R4, 50
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 51
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 52
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 53
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 54
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 55
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 56
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        MOV R4, 57
        LOAD R5, [R4]
        ADD R10, R10, R5
        
        # Store sum to output location
        MOV R6, 100
        STORE R10, [R6]
        
    skip_sum:
        JOIN
        RET
    """
    return Program.from_assembly(assembly)


def run_shared_memory_demo():
    """Run the shared memory demonstration."""
    gpu = GPU(precision='float32')
    
    program = create_shared_memory_program()
    gpu.load_program(program)
    
    # Initialize input data: values 1-8
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    gpu.global_memory.load_array(input_data, start_address=0)
    
    # Launch with 1 block of 8 threads
    gpu.launch(blocks=1, threads_per_block=8, threads_per_warp=8)
    
    console.print("[bold cyan]Shared Memory Example[/bold cyan]")
    console.print("=" * 60)
    console.print()
    console.print("This example demonstrates shared memory usage for reduction.")
    console.print()
    console.print("[yellow]Input values:[/yellow]", input_data)
    console.print(f"[yellow]Expected sum:[/yellow] {sum(input_data)}")
    console.print()
    
    # Run with visualization
    cycle_count = 0
    def show_progress(gpu):
        nonlocal cycle_count
        cycle_count = gpu.clock_cycle
        if gpu.clock_cycle % 5 == 0:
            console.print(f"  Cycle {gpu.clock_cycle}...", end="\r")
    
    gpu.run(render_callback=show_progress)
    
    console.print(f"\nCompleted in [bold]{cycle_count}[/bold] cycles")
    console.print()
    
    # Read result
    result = gpu.global_memory.read(100)
    console.print(f"[green]Result (sum):[/green] {int(result)}")
    
    expected = sum(input_data)
    if int(result) == expected:
        console.print("[green]✓ Correct![/green]")
    else:
        console.print(f"[red]✗ Expected {expected}[/red]")


if __name__ == "__main__":
    run_shared_memory_demo()
