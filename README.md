# Waterfall: Cycle-Accurate GPU Simulator

Waterfall is a Python-based, cycle-accurate GPU simulator designed for educational purposes. It implements a simplified SIMT (Single Instruction, Multiple Threads) architecture similar to CUDA, allowing users to understand the internal workings of a GPU, including warp scheduling, branch divergence, memory hierarchies, and instruction pipelining.

## 🚀 Features

- **SIMT Architecture**: Simulates the hierarchy of Grid -> Block -> Warp -> Thread.
- **Cycle-Accurate Execution**: Tracks execution cycle-by-cycle, including instruction latencies and pipeline stalls.
- **Warp Scheduling**: Implements a round-robin warp scheduler with configurable limits (default: 2 warps/cycle).
- **Branch Divergence**: Accurately handles divergent control flow with mask stacks and reconvergence points (`JOIN` opcode).
- **Memory Hierarchy**:
  - **Global Memory**: High-latency off-chip memory simulation.
  - **Shared Memory**: Low-latency on-chip memory for inter-thread communication.
  - **Register File**: Per-thread register storage.
- **Python Kernel Compiler**: Includes a compiler that translates Python functions (with `for`/`while` loops) into Waterfall assembly.
- **Precision Modes**: Supports `float16`, `float32`, and `float64` precision modes to demonstrating numerical effects.
- **Visual Dashboard**: Real-time terminal-based visualization using `rich` to show register states, memory contents, and execution progress.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/waterfall.git
   cd waterfall
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

Run the simulator using the `main.py` entry point. The simulator comes with several built-in examples demonstrating different GPU concepts.

```bash
python main.py [command]
```

### Available Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `simple` | `vector_add` | Run a simple Vector Addition kernel. |
| `matrix` | `matmul` | Run a Matrix Multiplication example. |
| `branch` | `divergence` | Demonstrate branch divergence handling. |
| `shared` | `reduction` | Demonstrate shared memory usage with parallel reduction. |
| `precision` | `fp16` | Compare Float16, Float32, and Float64 precision. |
| `memory` | `coalesced` | Visualize coalesced vs. strided memory access patterns. |
| `kernels` | `compile` | Demonstrate the Python-to-Assembly kernel compiler. |
| `scheduling`| `warps` | Visualize round-robin warp scheduling execution. |

### Example: Running Vector Addition

```bash
python main.py simple
```

This will launch the dashboard showing the value of registers (`R1`, `R2`, `R3`) for each thread as they perform the addition `C[i] = A[i] + B[i]`.

## 🏗️ Architecture

### Core Components (`pygpu/core/`)

- **GPU (`gpu.py`)**: Top-level controller. Managing memory allocation and kernel launches.
- **Streaming Multiprocessor (SM) (`sm.py`)**: The core execution unit. It manages resident blocks and schedules warps for execution.
- **Block (`block.py`)**: A group of threads that can communicate via shared memory.
- **Warp (`warp.py`)**: A collection of threads (simulated size: 4) that execute in lock-step. Handles the active mask for divergence.
- **Thread (`thread.py`)**: Main execution context containing PC (Program Counter) and Registers.
- **Instruction (`instruction.py`)**: Defines the ISA (Instruction Set Architecture) and opcode behavior.

### Instruction Set (ISA)

The simulator uses a custom assembly language. Key instructions include:

- **Arithmetic**: `ADD`, `SUB`, `MUL`, `DIV`
- **Memory**: `LOAD` (Global -> Reg), `STORE` (Reg -> Global), `LDS` (Shared -> Reg), `STS` (Reg -> Shared)
- **Control Flow**: `BRA` (Branch), `BEQ` (Branch if Equal), `BNE` (Branch Not Equal), `JMP` (Unconditional Jump), `JOIN` (Reconvergence point)
- **Synchronization**: `SYNC` (Block-wide barrier)

## 📂 Project Structure

```
.
├── bugs.md               # Tracking of known issues and fixes
├── examples/             # Example kernels and demonstrations
│   ├── branch_divergence.py
│   ├── matrix_multiply.py
│   ├── memory_patterns.py
│   ├── precision_demo.py
│   ├── python_kernels.py
│   ├── shared_memory.py
│   ├── vector_add.py
│   └── warp_scheduling.py
├── main.py               # CLI Entry point
├── pygpu/                # Simulator Source Code
│   ├── core/             # Internal architecture components
│   ├── ui/               # Visualization code
│   └── gpu.py            # Public API Class
└── requirements.txt      # Python dependencies
```

## 📝 Kernel Compiler

You can write kernels in Python and compile them to Waterfall assembly using the `KernelCompiler`.

**Supported Python constructs:**
- `for` loops (using `range`)
- `while` loops
- Basic arithmetic operators
- `if`/`else` conditionals

**Example Python Kernel:**
```python
def vector_add_kernel(block_id, thread_id):
    idx = block_id * 8 + thread_id
    if idx < 64:
        reg1 = global_memory[idx]      # Load A[idx]
        reg2 = global_memory[idx + 64] # Load B[idx]
        res = reg1 + reg2
        global_memory[idx + 128] = res # Store C[idx]
```

## 🤝 Contributing

Contributions are welcome! Please examine `bugs.md` for historical fixes and notes on the implementation details. If you find a bug, please detail it in an issue or PR.
