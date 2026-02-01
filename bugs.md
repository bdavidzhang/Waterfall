# Bugs and Issues in PyGPU Implementation

## 1. Incorrect Branch Divergence Reconvergence
**Severity: Critical**

The current implementation of branch divergence execution handles `IF/ELSE` blocks by executing one path (the "Taken" path) fully before switching to the other path. The condition for switching back (in `Warp._check_reconvergence`) is only met when the active threads are `FINISHED` (or waiting).

**Issue:** If there is common code *after* the `IF/ELSE` block, the first set of threads will continue to execute this common code until the end of the kernel. Then, the second set of threads will be restored, execute their branch, and *also* execute the common code.

**Consequence:** Code following a divergent branch is executed multiple times (once per path taken), leading to incorrect results for any operation that is not idempotent (e.g., `x += 1` in common code will happen twice).

**Suggested Fix:** The compiler needs to identify the Immediate Post-Dominator (IPDOM) of the branch — the point where control flows merge. A special instruction (e.g., `JOIN` or `POP`) should be inserted at this label. The divergence stack should pop when threads reach this IPDOM, not when they finish the kernel.

### Claude note:
**Fixed in:** [instruction.py](pygpu/core/instruction.py), [warp.py](pygpu/core/warp.py), [gpu.py](pygpu/gpu.py)

**What was done:**
1. Added a new `JOIN` opcode to the ISA in `instruction.py` to mark reconvergence points
2. Rewrote `_handle_branch()` in `warp.py` to find the JOIN instruction as the reconvergence point using `_find_reconvergence_point()`
3. Added `_handle_join()` method to properly handle reconvergence - it executes the TAKEN path, then switches to NOT-TAKEN path, then merges all threads
4. Added `executing_taken` field to `DivergenceStackEntry` to track which path is currently executing
5. Modified `_compile_if()` in the `KernelCompiler` to emit a `JOIN` instruction at the end of if/else blocks instead of a `NOP`
6. Added `_restore_all_threads()` helper method to restore thread masks after both paths complete 


## 2. Barrier Synchronization Deadlock
**Severity: Critical**

`Block.step` implements synchronization barriers (`__syncthreads()` / `OpCode.SYNC`) by checking if `at_sync == total_active`. However, `total_active` counts all threads that are not `FINISHED`, regardless of whether they are currently `MASKED` due to divergence.

**Issue:** If a `SYNC` instruction is placed in common code, but the warp is currently divergent (some threads are masked, waiting on the divergence stack), the `MASKED` threads cannot execute and thus cannot reach the `SYNC` instruction. The `ACTIVE` threads reach `SYNC` and wait. The block waits for `total_active` threads to reach `SYNC`. Since `MASKED` threads never arrive, the block deadlocks.

**Suggested Fix:**
1.  Ban `SYNC` inside divergent control flow (standard GPU behavior).
2.  If `SYNC` is in common code, the reconvergence mechanism (Fix #1) must ensure threads reconverge *before* hitting `SYNC`.
3.  `_check_barrier` should only count threads involved in the barrier if they are eligible to participate.

### Claude note:
**Fixed in:** [block.py](pygpu/core/block.py)

**What was done:**
1. Rewrote `_check_barrier()` to only count **ACTIVE** threads (not MASKED threads) as eligible for barrier synchronization
2. Added check for warp divergence state via `warp._divergence_stack`
3. Now only threads that are both `ACTIVE` and have `_active_mask[i] == True` are counted in `total_eligible`
4. MASKED threads (those in divergent branches) are explicitly skipped from the barrier count
5. This prevents deadlock because the barrier now only waits for threads that can actually reach it

## 3. Memory Latency is Not Simulated
**Severity: Major (Fidelity)**

The `GlobalMemory` class defines `LATENCY_CYCLES = 100`, but this constant is never used to delay execution.

**Issue:** `Thread.execute` performs `OpCode.LOAD` and `OpCode.STORE` operations instantaneously. This fails to simulate the primary bottleneck of GPU programming (memory latency).

**Suggested Fix:**
*   `OpCode.LOAD` should not return a value immediately. It should set the thread status to `WAITING` and initiate a "Memory Request".
*   The `GPU.tick()` loop should process memory requests, decrementing a counter for each request.
*   When the counter reaches zero, the value is written to the register and the thread status is set back to `ACTIVE`.

### Claude note:
**Fixed in:** [thread.py](pygpu/core/thread.py), [warp.py](pygpu/core/warp.py)

**What was done:**
1. Added `_pending_load` field to `Thread` class to track in-flight memory requests with `{dst_reg, address, cycles_remaining}`
2. Modified `OpCode.LOAD` handling in `Thread.execute()` to:
   - Set `status = ThreadStatus.WAITING` instead of completing immediately
   - Store the pending load info with `cycles_remaining = GlobalMemory.LATENCY_CYCLES` (100 cycles)
   - Return `False` to indicate the thread is stalled
3. Added logic at the start of `execute()` to process pending loads:
   - Decrements `cycles_remaining` each cycle
   - When counter reaches 0, performs the actual memory read and sets thread back to `ACTIVE`
4. Modified `Warp.step()` to detect and handle waiting threads:
   - Checks for any threads with `WAITING` status
   - Calls `execute()` on waiting threads to process their pending memory operations
   - Tracks `cycles_stalled` for visualization

## 4. Unrealistic Warp Scheduling
**Severity: Minor (Fidelity)**

The `StreamingMultiprocessor` executes one instruction for *every* warp in *every* active block during a single cycle.

**Issue:** Real SMs have limited dispatch units and execution resources (e.g., 4 schedulers, each issuing 1 inst/cycle). Simulating infinite issue width dramatically underestimates cycle counts and hides resource contention.

**Suggested Fix:** `SM.step()` should limit the number of warps that can issue an instruction per cycle (e.g., `MAX_WARPS_PER_CYCLE = 2`). It should select warps (e.g., Round Robin) to execute.

### Claude note:
**Fixed in:** [sm.py](pygpu/core/sm.py)

**What was done:**
1. Added `MAX_WARPS_PER_CYCLE = 2` class constant to limit instruction issue per cycle
2. Added `_next_warp_index` field for round-robin scheduling state
3. Rewrote `step()` to:
   - Collect all ready (non-finished, non-stalled) warps from all active blocks
   - Use round-robin selection starting from `_next_warp_index` to pick up to `MAX_WARPS_PER_CYCLE` warps
   - Only those selected warps execute in the current cycle
   - Update `_next_warp_index` for fairness across cycles
4. Stalled warps (waiting for memory) are still processed to update their pending operations

## 5. Compiler Limitations
**Severity: Moderate**

The `KernelCompiler` is very limited:
*   **No Loops:** It supports `If` but not `For` or `While`, making it impossible to write meaningful kernels (like matrix multiplication with loops).
*   **No Function Calls:** Cannot call other functions.
*   **No Stack:** There is no local stack implementation for function calls or spilling registers.

**Suggested Fix:** Implement support for `ast.For` and `ast.While` in `KernelCompiler` using jumps and labels.

### Claude note:
**Fixed in:** [gpu.py](pygpu/gpu.py)

**What was done:**
1. Added `ast.For` and `ast.While` handling in `_compile_stmt()`
2. Implemented `_compile_for()` method:
   - Supports `for i in range(end)`, `for i in range(start, end)`, and `for i in range(start, end, step)`
   - Allocates register for loop variable
   - Generates loop start label, condition check (`CMP_LT`), conditional branch to exit, body, increment, and jump back to start
3. Implemented `_compile_while()` method:
   - Generates loop start label, condition evaluation, conditional branch to exit, body, and jump back to start
4. Both loops use proper labels (`loop_start`, `loop_end`, `while_start`, `while_end`) for control flow
5. **Note:** Function calls and stack-based operations are still not supported (would require significant additional work for call stack, register spilling, etc.)

## 6. Thread Execution Safety
**Severity: Minor**

`Thread.execute` directly modifies registers. If an exception occurs (e.g., division by zero handled by numpy, or index out of bounds), the simulator might crash or behave unpredictably. `GlobalMemory` raises `IndexError` on OOB access, which would crash the simulator loop.

**Suggested Fix:** Wrap execution in try/except blocks to catch runtime errors and mark the thread/block as `ERROR` or explicitly handle faults.

### Claude note:
**Fixed in:** [thread.py](pygpu/core/thread.py)

**What was done:**
1. Added `ERROR` status to `ThreadStatus` enum
2. Added `error_message` field to `Thread` class to store error details
3. Wrapped the entire instruction execution logic in a `try/except` block
4. On any exception (IndexError, arithmetic errors, etc.):
   - Set `status = ThreadStatus.ERROR`
   - Store the error message with PC information in `error_message`
   - Return `False` to stop execution
5. Added early return check for `ERROR` status at the start of `execute()` to prevent further execution
6. Memory read errors during latency simulation are also caught and handled
