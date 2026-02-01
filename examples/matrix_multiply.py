"""
Matrix Multiplication Example

A simplified matrix multiplication kernel to demonstrate
more complex GPU workloads and memory access patterns.

This uses a naive algorithm where each thread computes one element
of the output matrix.
"""

from rich.console import Console

from pygpu import GPU
from pygpu.core.instruction import Program

console = Console()


def create_matrix_mul_program(n: int = 4) -> Program:
    """
    Create a simplified matrix multiplication program.
    
    Computes C = A * B for N x N matrices.
    
    Memory layout:
    - [0:N*N] = Matrix A (row-major)
    - [N*N:2*N*N] = Matrix B (row-major)
    - [2*N*N:3*N*N] = Matrix C (result)
    
    Each thread computes one element of C:
    C[row][col] = sum(A[row][k] * B[k][col] for k in range(N))
    """
    # For simplicity, we'll do a 2x2 matrix multiply
    # Each of 4 threads computes one output element
    
    assembly = f"""
        # Get thread ID (0-3 for 2x2 matrix)
        THREAD_ID R0
        
        # Calculate row = thread_id / N (integer division)
        MOV R1, {n}
        IDIV R2, R0, R1
        
        # Calculate col = thread_id % N  
        MOD R3, R0, R1      # R3 = col
        
        # Initialize accumulator R10 = 0
        MOV R10, 0
        
        # Loop counter k = 0
        MOV R4, 0
        
    loop_start:
        # Check if k < N
        CMP_LT P0, R4, R1
        BRA_N loop_end, P0
        
        # Calculate address of A[row][k]
        # addr_a = row * N + k = R2 * N + R4
        MUL R5, R2, R1
        ADD R5, R5, R4
        
        # Load A[row][k]
        LOAD R6, [R5]
        
        # Calculate address of B[k][col]
        # addr_b = N*N + k * N + col = {n*n} + R4 * N + R3
        MUL R7, R4, R1
        ADD R7, R7, R3
        MOV R8, {n*n}
        ADD R7, R7, R8
        
        # Load B[k][col]
        LOAD R8, [R7]
        
        # Accumulate: R10 += A[row][k] * B[k][col]
        MUL R9, R6, R8
        ADD R10, R10, R9
        
        # k++
        MOV R11, 1
        ADD R4, R4, R11
        JMP loop_start
        
    loop_end:
        # Calculate address of C[row][col]
        # addr_c = 2*N*N + row * N + col
        MUL R5, R2, R1
        ADD R5, R5, R3
        MOV R6, {2*n*n}
        ADD R5, R5, R6
        
        # Store result
        STORE R10, [R5]
        
        RET
    """
    return Program.from_assembly(assembly)


def run_matrix_multiply():
    """Run the matrix multiplication example."""
    N = 2  # 2x2 matrices for simplicity
    
    # Create GPU
    gpu = GPU(precision='float32')
    
    # Load program
    program = create_matrix_mul_program(N)
    gpu.load_program(program)
    
    # Initialize matrices
    # A = [[1, 2], [3, 4]]
    matrix_a = [1, 2, 3, 4]
    gpu.global_memory.load_array(matrix_a, start_address=0)
    
    # B = [[5, 6], [7, 8]]
    matrix_b = [5, 6, 7, 8]
    gpu.global_memory.load_array(matrix_b, start_address=N*N)
    
    # Launch with 1 block of 4 threads (one per output element)
    gpu.launch(blocks=1, threads_per_block=4, threads_per_warp=4)
    
    console.print("[bold cyan]Matrix Multiplication Example[/bold cyan]")
    console.print("=" * 50)
    console.print()
    console.print("[yellow]Matrix A:[/yellow]")
    console.print(f"  | {matrix_a[0]:3} {matrix_a[1]:3} |")
    console.print(f"  | {matrix_a[2]:3} {matrix_a[3]:3} |")
    console.print()
    console.print("[yellow]Matrix B:[/yellow]")
    console.print(f"  | {matrix_b[0]:3} {matrix_b[1]:3} |")
    console.print(f"  | {matrix_b[2]:3} {matrix_b[3]:3} |")
    console.print()
    
    # Run simulation
    cycles = gpu.run()
    console.print(f"Completed in [bold]{cycles}[/bold] cycles")
    console.print()
    
    # Read result
    result = gpu.global_memory.read_array(2*N*N, N*N)
    console.print("[green]Result C = A × B:[/green]")
    console.print(f"  | {int(result[0]):3} {int(result[1]):3} |")
    console.print(f"  | {int(result[2]):3} {int(result[3]):3} |")
    
    # Verify
    # C[0][0] = 1*5 + 2*7 = 19
    # C[0][1] = 1*6 + 2*8 = 22
    # C[1][0] = 3*5 + 4*7 = 43
    # C[1][1] = 3*6 + 4*8 = 50
    expected = [19, 22, 43, 50]
    console.print()
    console.print("[dim]Expected:[/dim]")
    console.print(f"  | {expected[0]:3} {expected[1]:3} |")
    console.print(f"  | {expected[2]:3} {expected[3]:3} |")
    
    if list(map(int, result)) == expected:
        console.print("\n[bold green]✓ Verification PASSED![/bold green]")
    else:
        console.print(f"\n[bold red]✗ Verification FAILED![/bold red]")


if __name__ == "__main__":
    run_matrix_multiply()
