"""
Memory Access Patterns Example

This example demonstrates different memory access patterns and their
impact on GPU performance. We visualize the memory access heatmap
to show coalesced vs. strided access patterns.
"""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel

from pygpu import GPU
from pygpu.core.instruction import Program

console = Console()


def create_coalesced_access_program(n_threads: int = 8) -> Program:
    """
    Create a program with coalesced (sequential) memory access.
    
    Thread i accesses memory address i.
    This is the optimal access pattern for GPUs.
    """
    assembly = """
        THREAD_ID R0
        
        # Load from address = thread_id (coalesced)
        LOAD R1, [R0]
        
        # Multiply by 2
        MOV R2, 2
        MUL R3, R1, R2
        
        # Store to output (coalesced)
        MOV R4, 100
        ADD R4, R4, R0
        STORE R3, [R4]
        
        RET
    """
    return Program.from_assembly(assembly)


def create_strided_access_program(stride: int = 8) -> Program:
    """
    Create a program with strided memory access.
    
    Thread i accesses memory address i * stride.
    This is inefficient and causes multiple memory transactions.
    """
    assembly = f"""
        THREAD_ID R0
        
        # Load from address = thread_id * stride (strided)
        MOV R1, {stride}
        MUL R2, R0, R1
        LOAD R3, [R2]
        
        # Multiply by 2
        MOV R4, 2
        MUL R5, R3, R4
        
        # Store with same stride
        MOV R6, 100
        MUL R7, R0, R1
        ADD R7, R7, R6
        STORE R5, [R7]
        
        RET
    """
    return Program.from_assembly(assembly)


def create_random_access_program() -> Program:
    """
    Create a program with pseudo-random memory access.
    
    Thread i accesses address based on a simple hash.
    This pattern is common in hash tables, graph algorithms, etc.
    """
    assembly = """
        THREAD_ID R0
        
        # Pseudo-random address: (thread_id * 7 + 3) % 16
        MOV R1, 7
        MUL R2, R0, R1
        MOV R3, 3
        ADD R2, R2, R3
        MOV R4, 16
        MOD R2, R2, R4
        
        # Load from random address
        LOAD R5, [R2]
        
        # Process
        MOV R6, 1
        ADD R7, R5, R6
        
        # Store back
        STORE R7, [R2]
        
        RET
    """
    return Program.from_assembly(assembly)


def visualize_memory_heatmap(gpu: GPU, start: int, end: int, title: str):
    """Visualize memory access pattern as a heatmap."""
    heatmap = gpu.get_memory_heatmap(start, end)
    max_access = max(heatmap) if heatmap else 1
    
    # Create visual representation
    line = Text()
    for count in heatmap:
        if count == 0:
            line.append("░", style="dim white")
        elif count == 1:
            line.append("▓", style="green")
        elif count <= max_access // 2:
            line.append("█", style="yellow")
        else:
            line.append("█", style="red")
    
    console.print(f"[bold]{title}[/bold]")
    console.print(f"Addresses {start}-{end-1}:")
    console.print(line)
    console.print()


def run_memory_patterns_demo():
    """Run the memory access patterns demonstration."""
    console.print("[bold cyan]Memory Access Patterns Example[/bold cyan]")
    console.print("=" * 70)
    console.print()
    console.print("This example shows how different memory access patterns affect")
    console.print("GPU performance and memory utilization.")
    console.print()
    
    # Legend
    legend = Text()
    legend.append("Legend: ")
    legend.append("░", style="dim white")
    legend.append("=No access  ")
    legend.append("▓", style="green")
    legend.append("=1 access  ")
    legend.append("█", style="yellow")
    legend.append("=Few accesses  ")
    legend.append("█", style="red")
    legend.append("=Many accesses")
    console.print(Panel(legend, title="Memory Heatmap Legend"))
    console.print()
    
    n_threads = 8
    
    # Test 1: Coalesced access
    console.print("[bold green]1. Coalesced Access (Optimal)[/bold green]")
    console.print("   Thread i reads from address i - sequential, efficient")
    console.print()
    
    gpu1 = GPU(precision='float32')
    gpu1.global_memory.load_array([float(i) for i in range(16)], 0)
    gpu1.load_program(create_coalesced_access_program())
    gpu1.launch(blocks=1, threads_per_block=n_threads)
    gpu1.run()
    
    visualize_memory_heatmap(gpu1, 0, 16, "Input reads:")
    visualize_memory_heatmap(gpu1, 100, 116, "Output writes:")
    console.print(f"   Cycles: {gpu1.clock_cycle}")
    console.print()
    
    # Test 2: Strided access
    console.print("[bold yellow]2. Strided Access (Inefficient)[/bold yellow]")
    console.print("   Thread i reads from address i*8 - non-contiguous, slower")
    console.print()
    
    gpu2 = GPU(precision='float32')
    gpu2.global_memory.load_array([float(i) for i in range(128)], 0)
    gpu2.load_program(create_strided_access_program(stride=8))
    gpu2.launch(blocks=1, threads_per_block=n_threads)
    gpu2.run()
    
    visualize_memory_heatmap(gpu2, 0, 64, "Input reads (strided):")
    visualize_memory_heatmap(gpu2, 100, 164, "Output writes (strided):")
    console.print(f"   Cycles: {gpu2.clock_cycle}")
    console.print()
    
    # Test 3: Random access
    console.print("[bold red]3. Random Access (Worst Case)[/bold red]")
    console.print("   Threads access pseudo-random addresses - unpredictable")
    console.print()
    
    gpu3 = GPU(precision='float32')
    gpu3.global_memory.load_array([float(i) for i in range(16)], 0)
    gpu3.load_program(create_random_access_program())
    gpu3.launch(blocks=1, threads_per_block=n_threads)
    gpu3.run()
    
    visualize_memory_heatmap(gpu3, 0, 16, "Random access pattern:")
    console.print(f"   Cycles: {gpu3.clock_cycle}")
    console.print()
    
    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print()
    console.print("  [green]Coalesced[/green]: All threads access consecutive addresses")
    console.print("                → Memory controller combines into one transaction")
    console.print()
    console.print("  [yellow]Strided[/yellow]:   Threads access with fixed gaps")
    console.print("                → Multiple memory transactions needed")
    console.print()
    console.print("  [red]Random[/red]:     Unpredictable access pattern")
    console.print("                → Poor cache utilization, many transactions")
    console.print()
    console.print("  [bold]Key Insight:[/bold] Optimize memory access patterns for GPU performance!")


if __name__ == "__main__":
    run_memory_patterns_demo()
