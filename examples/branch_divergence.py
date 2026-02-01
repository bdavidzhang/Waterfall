"""
Branch Divergence Example

This example demonstrates what happens when threads in a warp
take different branches in an if/else statement.

This is the most visually interesting aspect of SIMT execution!
"""

from rich.console import Console
from rich.text import Text

from pygpu import GPU
from pygpu.core.instruction import Program

console = Console()


def create_divergent_program() -> Program:
    """
    Create a program that causes branch divergence.
    
    Each thread checks if its ID is even or odd:
    - Even threads: result = thread_id * 2
    - Odd threads: result = thread_id * 3
    """
    assembly = """
        # Get thread ID
        THREAD_ID R0
        
        # R1 = thread_id % 2 (check if even/odd)
        MOV R1, 2
        MOD R2, R0, R1
        
        # Compare: P0 = (R2 == 0)
        MOV R3, 0
        CMP_EQ P0, R2, R3
        
        # Branch based on even/odd
        BRA even_path, P0
        
    odd_path:
        # Odd thread: result = thread_id * 3
        MOV R4, 3
        MUL R5, R0, R4
        JMP store_result
        
    even_path:
        # Even thread: result = thread_id * 2
        MOV R4, 2
        MUL R5, R0, R4
        
    store_result:
        # Store result at address = thread_id
        STORE R5, [R0]
        RET
    """
    return Program.from_assembly(assembly)


def run_divergence_demo():
    """Run the branch divergence demonstration."""
    gpu = GPU(precision='float32')
    
    program = create_divergent_program()
    gpu.load_program(program)
    
    # Launch with 1 block of 8 threads
    gpu.launch(blocks=1, threads_per_block=8, threads_per_warp=8)
    
    console.print("[bold cyan]Branch Divergence Example[/bold cyan]")
    console.print("=" * 60)
    console.print()
    console.print("Each thread checks if its ID is even or odd:")
    console.print("  - [green]Even threads[/green]: result = thread_id * 2")
    console.print("  - [yellow]Odd threads[/yellow]:  result = thread_id * 3")
    console.print()
    console.print("In SIMT execution, threads that take different branches")
    console.print("cause 'divergence' - they execute serially, not in parallel.")
    console.print()
    
    # Color legend
    legend = Text()
    legend.append("Legend: ")
    legend.append("█", style="bold green")
    legend.append("=Active  ")
    legend.append("░", style="dim white")
    legend.append("=Masked  ")
    legend.append("▪", style="bold blue")
    legend.append("=Finished")
    console.print(legend)
    console.print("=" * 60)
    
    # Run with visualization callback
    def show_state(gpu):
        state = gpu.get_state()
        cycle = state['cycle']
        
        line = Text()
        line.append(f"\nCycle {cycle:2d}: ", style="dim")
        
        for block in state['blocks']:
            for warp in block['thread_grid']:
                for s in warp:
                    if s == "ACTIVE":
                        line.append("█ ", style="bold green")
                    elif s == "MASKED":
                        line.append("░ ", style="dim white")
                    elif s == "FINISHED":
                        line.append("▪ ", style="bold blue")
                    else:
                        line.append("? ", style="red")
        
        console.print(line, end="")
        
        # Show which threads are active
        for block in state['blocks']:
            for warp in block['thread_grid']:
                active_ids = [str(i) for i, s in enumerate(warp) if s == "ACTIVE"]
                if active_ids:
                    console.print(f"  [dim][active: {', '.join(active_ids)}][/dim]", end="")
    
    gpu.run(render_callback=show_state)
    
    console.print("\n\n" + "=" * 60)
    console.print("[bold]Results:[/bold]")
    
    result = gpu.global_memory.read_array(0, 8)
    for i, val in enumerate(result):
        even_odd = "even" if i % 2 == 0 else "odd"
        expected = i * 2 if i % 2 == 0 else i * 3
        if int(val) == expected:
            console.print(f"  Thread {i} ({even_odd}): {int(val):3d} (expected {expected}) [green]✓[/green]")
        else:
            console.print(f"  Thread {i} ({even_odd}): {int(val):3d} (expected {expected}) [red]✗[/red]")


def create_nested_divergence_program() -> Program:
    """
    Create a program with nested divergence.
    
    Threads branch based on:
    1. thread_id % 4 == 0 (quarter of threads)
    2. thread_id % 2 == 0 (remaining half)
    3. Rest (remaining quarter)
    """
    assembly = """
        THREAD_ID R0
        
        # Check if thread_id % 4 == 0
        MOV R1, 4
        MOD R2, R0, R1
        MOV R3, 0
        CMP_EQ P0, R2, R3
        BRA quarter_path, P0
        
        # Check if thread_id % 2 == 0
        MOV R1, 2
        MOD R2, R0, R1
        CMP_EQ P1, R2, R3
        BRA half_path, P1
        
    rest_path:
        # thread_id % 2 != 0 and thread_id % 4 != 0
        MOV R5, 1
        JMP store
        
    half_path:
        # thread_id % 2 == 0 but thread_id % 4 != 0
        MOV R5, 2
        JMP store
        
    quarter_path:
        # thread_id % 4 == 0
        MOV R5, 4
        
    store:
        STORE R5, [R0]
        RET
    """
    return Program.from_assembly(assembly)


if __name__ == "__main__":
    run_divergence_demo()
