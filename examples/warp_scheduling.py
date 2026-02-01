"""
Warp Scheduling Visualization Example

This example demonstrates how the warp scheduler distributes work
across multiple warps and blocks, showing the round-robin scheduling
and resource constraints.
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
import time

from pygpu import GPU
from pygpu.core.instruction import Program

console = Console()


def create_workload_program(iterations: int = 10) -> Program:
    """
    Create a program that does some work (multiple iterations).
    
    This helps visualize how warps are scheduled over time.
    """
    assembly = f"""
        THREAD_ID R0
        MOV R1, 0           # Counter
        MOV R2, {iterations}  # Max iterations
        
    loop:
        # Check if counter < max
        CMP_LT P0, R1, R2
        BRA_N done, P0
        
        # Do some work: R3 = R3 + thread_id
        ADD R3, R3, R0
        
        # Increment counter
        MOV R4, 1
        ADD R1, R1, R4
        JMP loop
        
    done:
        JOIN
        # Store result
        STORE R3, [R0]
        RET
    """
    return Program.from_assembly(assembly)


def run_warp_scheduling_demo():
    """Run the warp scheduling visualization."""
    console.print("[bold cyan]Warp Scheduling Visualization[/bold cyan]")
    console.print("=" * 70)
    console.print()
    console.print("This example shows how the SM schedules warps for execution.")
    console.print("With MAX_WARPS_PER_CYCLE=2, only 2 warps can issue per cycle.")
    console.print()
    
    # Create GPU with 1 SM
    gpu = GPU(precision='float32', num_sms=1)
    
    program = create_workload_program(iterations=5)
    gpu.load_program(program)
    
    # Launch 2 blocks with 16 threads each = 4 warps total (2 per block)
    gpu.launch(blocks=2, threads_per_block=16, threads_per_warp=8)
    
    console.print("[yellow]Configuration:[/yellow]")
    console.print(f"  - 2 blocks × 16 threads/block = 32 threads")
    console.print(f"  - 8 threads/warp = 4 warps total")
    console.print(f"  - MAX_WARPS_PER_CYCLE = 2")
    console.print()
    
    # Track scheduling history
    schedule_history = []
    
    def track_schedule(gpu):
        state = gpu.get_state()
        cycle_info = {
            'cycle': state['cycle'],
            'warps': []
        }
        
        for sm_state in state['sms']:
            pass  # SM-level tracking
        
        for block in state['blocks']:
            block_id = block['block_id']
            for warp_idx, warp_states in enumerate(block['thread_grid']):
                active = sum(1 for s in warp_states if s == 'ACTIVE')
                masked = sum(1 for s in warp_states if s == 'MASKED')
                finished = sum(1 for s in warp_states if s == 'FINISHED')
                waiting = sum(1 for s in warp_states if s == 'WAITING')
                
                cycle_info['warps'].append({
                    'block': block_id,
                    'warp': warp_idx,
                    'active': active,
                    'masked': masked,
                    'finished': finished,
                    'waiting': waiting,
                })
        
        schedule_history.append(cycle_info)
    
    # Run simulation
    gpu.run(render_callback=track_schedule)
    
    # Display schedule history
    console.print("[yellow]Execution Timeline:[/yellow]")
    console.print()
    
    # Show first 20 cycles
    table = Table(title="Warp Activity per Cycle")
    table.add_column("Cycle", justify="center", width=6)
    table.add_column("B0-W0", justify="center", width=8)
    table.add_column("B0-W1", justify="center", width=8)
    table.add_column("B1-W0", justify="center", width=8)
    table.add_column("B1-W1", justify="center", width=8)
    
    for cycle_info in schedule_history[:25]:
        row = [str(cycle_info['cycle'])]
        
        for warp_info in cycle_info['warps']:
            if warp_info['finished'] == 8:
                cell = "[blue]DONE[/blue]"
            elif warp_info['active'] > 0:
                cell = f"[green]●{warp_info['active']}[/green]"
            elif warp_info['waiting'] > 0:
                cell = f"[yellow]◐{warp_info['waiting']}[/yellow]"
            elif warp_info['masked'] > 0:
                cell = f"[dim]○{warp_info['masked']}[/dim]"
            else:
                cell = "[dim]-[/dim]"
            row.append(cell)
        
        # Pad if fewer warps
        while len(row) < 5:
            row.append("-")
        
        table.add_row(*row)
    
    console.print(table)
    
    # Legend
    console.print()
    legend = Text()
    legend.append("Legend: ")
    legend.append("●N", style="green")
    legend.append("=N active  ")
    legend.append("◐N", style="yellow")
    legend.append("=N waiting  ")
    legend.append("○N", style="dim")
    legend.append("=N masked  ")
    legend.append("DONE", style="blue")
    legend.append("=finished")
    console.print(legend)
    
    console.print()
    console.print(f"[green]Total cycles: {gpu.clock_cycle}[/green]")
    console.print()
    
    # Verify results
    console.print("[yellow]Results (thread_id × iterations):[/yellow]")
    results = gpu.global_memory.read_array(0, 16)
    for i, val in enumerate(results):
        expected = i * 5  # iterations = 5, so thread_id * 5
        status = "[green]✓[/green]" if int(val) == expected else "[red]✗[/red]"
        if i < 8:
            console.print(f"  Thread {i:2d} (Block 0): {int(val):3d} {status}")


def run_multi_sm_demo():
    """Demonstrate multi-SM execution."""
    console.print("\n[bold cyan]Multi-SM Execution[/bold cyan]")
    console.print("=" * 70)
    console.print()
    
    # Compare 1 SM vs 2 SMs
    for num_sms in [1, 2]:
        gpu = GPU(precision='float32', num_sms=num_sms)
        gpu.load_program(create_workload_program(iterations=5))
        gpu.launch(blocks=4, threads_per_block=8, threads_per_warp=8)
        
        cycles = gpu.run()
        
        console.print(f"  {num_sms} SM(s): {cycles} cycles")
    
    console.print()
    console.print("  [dim]More SMs = blocks distributed, faster completion[/dim]")


if __name__ == "__main__":
    run_warp_scheduling_demo()
    run_multi_sm_demo()
