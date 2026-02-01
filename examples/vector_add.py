"""
Vector Addition Example

This example demonstrates the classic GPU hello-world: adding two vectors.
Each thread computes one element of the result: C[i] = A[i] + B[i]
"""

from rich.console import Console
from rich.text import Text
from rich.table import Table

from pygpu import GPU
from pygpu.core.instruction import Program, Instruction, OpCode

console = Console()


def create_vector_add_program(threads_per_block: int = 8) -> Program:
    """
    Create a vector addition program in assembly.
    
    Memory layout:
    - [0:100] = Vector A
    - [100:200] = Vector B
    - [200:300] = Result C
    """
    # Assembly program for: C[global_id] = A[global_id] + B[global_id]
    # global_id = block_id * threads_per_block + thread_id
    assembly = f"""
        # Get thread ID into R0 (local to block)
        THREAD_ID R0
        
        # Get block ID into R1
        BLOCK_ID R1
        
        # Compute global thread ID: R2 = block_id * threads_per_block + thread_id
        MOV R7, {threads_per_block}
        MUL R2, R1, R7
        ADD R2, R2, R0
        
        # Calculate address for A[global_id]
        # R3 = 0 + R2 (base of A is 0)
        MOV R3, R2
        
        # Load A[global_id] into R4
        LOAD R4, [R3]
        
        # Calculate address for B[global_id]
        # R5 = 100 + R2
        MOV R5, 100
        ADD R5, R5, R2
        
        # Load B[global_id] into R6
        LOAD R6, [R5]
        
        # R8 = A + B
        ADD R8, R4, R6
        
        # Calculate address for C[global_id]
        # R9 = 200 + R2
        MOV R9, 200
        ADD R9, R9, R2
        
        # Store result to C[global_id]
        STORE R8, [R9]
        
        # Return
        RET
    """
    return Program.from_assembly(assembly)


def vector_add_kernel(thread_id, global_mem):
    """
    Vector addition kernel (Python version).
    
    This can be compiled by the KernelCompiler.
    """
    # Load values from A and B
    idx = thread_id
    val_a = global_mem[idx]
    val_b = global_mem[idx + 100]
    
    # Compute sum
    result = val_a + val_b
    
    # Store to C
    global_mem[idx + 200] = result


def run_vector_add():
    """Run the vector addition example."""
    # Create GPU
    gpu = GPU(precision='float32')
    
    # Load the assembly program (with threads_per_block=8)
    program = create_vector_add_program(threads_per_block=8)
    gpu.load_program(program)
    
    # Initialize input data
    # Vector A: [0, 1, 2, 3, ...]
    vector_a = list(range(16))
    gpu.global_memory.load_array(vector_a, start_address=0)
    
    # Vector B: [100, 101, 102, 103, ...]
    vector_b = [100 + i for i in range(16)]
    gpu.global_memory.load_array(vector_b, start_address=100)
    
    # Launch kernel with 2 blocks, 8 threads each
    gpu.launch(blocks=2, threads_per_block=8)
    
    console.print("[bold cyan]Vector Addition Example[/bold cyan]")
    console.print("=" * 50)
    console.print(f"[yellow]Vector A:[/yellow] {vector_a}")
    console.print(f"[yellow]Vector B:[/yellow] {vector_b}")
    console.print()
    
    # Run the simulation
    cycles = gpu.run()
    
    console.print(f"Completed in [bold]{cycles}[/bold] cycles")
    console.print()
    
    # Read results
    result = gpu.global_memory.read_array(200, 16)
    result_ints = [int(x) for x in result]
    console.print(f"[green]Result C:[/green] {result_ints}")
    
    # Verify
    expected = [a + b for a, b in zip(vector_a, vector_b)]
    if result_ints == expected:
        console.print("\n[bold green]✓ Verification PASSED![/bold green]")
    else:
        console.print(f"\n[bold red]✗ Verification FAILED![/bold red]")
        console.print(f"  Expected: {expected}")


def run_vector_add_interactive():
    """Run the vector addition with step-by-step output."""
    from pygpu import GPU
    
    gpu = GPU(precision='float32')
    program = create_vector_add_program()
    gpu.load_program(program)
    
    # Initialize data
    vector_a = list(range(8))
    vector_b = [100 + i for i in range(8)]
    gpu.global_memory.load_array(vector_a, start_address=0)
    gpu.global_memory.load_array(vector_b, start_address=100)
    
    # Launch with 1 block of 8 threads
    gpu.launch(blocks=1, threads_per_block=8)
    
    console.print("[bold cyan]Step-by-step Vector Addition[/bold cyan]")
    console.print("=" * 60)
    console.print(f"[yellow]Vector A:[/yellow] {vector_a}")
    console.print(f"[yellow]Vector B:[/yellow] {vector_b}")
    console.print()
    
    # Color legend
    legend = Text()
    legend.append("Legend: ")
    legend.append("█", style="bold green")
    legend.append("=Active  ")
    legend.append("▪", style="bold blue")
    legend.append("=Finished  ")
    legend.append("Read", style="bold magenta")
    legend.append("  ")
    legend.append("Write", style="bold red")
    console.print(legend)
    console.print("=" * 60)
    
    def render_state(gpu):
        state = gpu.get_state()
        cycle = state['cycle']
        
        line = Text()
        line.append(f"\nCycle {cycle:2d}: ", style="dim")
        
        for block in state['blocks']:
            grid = block['thread_grid']
            line.append(f"Block {block['block_id']} ", style="cyan")
            for warp in grid:
                line.append("[")
                for i, s in enumerate(warp):
                    if s == "ACTIVE":
                        line.append("█", style="bold green")
                    elif s == "FINISHED":
                        line.append("▪", style="bold blue")
                    elif s == "MASKED":
                        line.append("░", style="dim")
                    else:
                        line.append("?", style="red")
                    if i < len(warp) - 1:
                        line.append(" ")
                line.append("] ")
        
        # Memory access info
        if state['memory_reads']:
            line.append(f" Read: {sorted(state['memory_reads'])}", style="magenta")
        if state['memory_writes']:
            line.append(f" Write: {sorted(state['memory_writes'])}", style="red")
        
        console.print(line)
    
    gpu.run(render_callback=render_state)
    
    console.print("\n" + "=" * 60)
    result = gpu.global_memory.read_array(200, 8)
    result_ints = [int(x) for x in result]
    console.print(f"[bold green]Result C = A + B:[/bold green] {result_ints}")


if __name__ == "__main__":
    run_vector_add()
    print("\n" + "=" * 50 + "\n")
    run_vector_add_interactive()
