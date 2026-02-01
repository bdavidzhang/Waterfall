"""
Block - A group of Warps that share Shared Memory.

In CUDA terminology, this is a "thread block" or "cooperative thread array (CTA)".
All threads in a block can synchronize with each other and share data via shared memory.
"""

from typing import List, Optional, Set

from pygpu.core.warp import Warp
from pygpu.core.thread import Thread, ThreadStatus
from pygpu.core.instruction import Program
from pygpu.core.memory import GlobalMemory, SharedMemory
from pygpu.core.alu import Precision


class Block:
    """
    A block of warps sharing shared memory.
    
    Blocks are the unit of work distribution in GPUs. Each block is assigned
    to a Streaming Multiprocessor (SM) for execution.
    """
    
    def __init__(
        self,
        block_id: int,
        num_threads: int,
        threads_per_warp: int = Warp.DEFAULT_SIZE,
        program: Optional[Program] = None,
        precision: Precision = Precision.FLOAT32,
        shared_memory_size: int = SharedMemory.DEFAULT_SIZE,
    ):
        self.block_id = block_id
        self.num_threads = num_threads
        self.threads_per_warp = threads_per_warp
        self.program = program
        self.precision = precision
        
        # Create shared memory for this block
        self.shared_memory = SharedMemory(shared_memory_size, precision)
        
        # Calculate number of warps needed
        self.num_warps = (num_threads + threads_per_warp - 1) // threads_per_warp
        
        # Create warps
        self.warps: List[Warp] = []
        for warp_idx in range(self.num_warps):
            # Calculate thread ID offset for this warp
            thread_id_offset = warp_idx * threads_per_warp
            
            # Last warp might have fewer threads
            warp_size = min(threads_per_warp, num_threads - thread_id_offset)
            
            warp = Warp(
                warp_id=warp_idx,
                block_id=block_id,
                num_threads=warp_size,
                program=program,
                precision=precision,
                thread_id_offset=thread_id_offset,
            )
            self.warps.append(warp)
        
        # Barrier synchronization tracking
        self._threads_at_barrier: Set[int] = set()
        self._barrier_active = False
        
        # Execution tracking
        self.cycles_executed: int = 0
    
    @property
    def is_finished(self) -> bool:
        """Check if all warps in the block have finished."""
        return all(warp.is_finished for warp in self.warps)
    
    @property
    def all_threads(self) -> List[Thread]:
        """Get a flat list of all threads in the block."""
        threads = []
        for warp in self.warps:
            threads.extend(warp.threads)
        return threads
    
    def step(self, global_memory: GlobalMemory) -> bool:
        """
        Execute one cycle for all warps in the block.
        
        Returns True if any warp made progress.
        """
        if self.is_finished:
            return False
        
        # Check for barrier synchronization
        if self._check_barrier():
            return True
        
        # Step each warp
        any_progress = False
        for warp in self.warps:
            if not warp.is_finished:
                progress = warp.step(global_memory, self.shared_memory)
                if progress:
                    any_progress = True
        
        self.cycles_executed += 1
        
        # Clear memory access tracking for next cycle
        self.shared_memory.clear_access_tracking()
        
        return any_progress
    
    def _check_barrier(self) -> bool:
        """
        Check and handle barrier synchronization.
        
        Returns True if we're waiting at a barrier.
        
        Note: SYNC should only be used in non-divergent code. If threads are
        divergent (masked), they cannot reach the barrier. We only count
        ACTIVE threads for barrier synchronization.
        """
        # Count threads at sync instruction (only ACTIVE threads can participate)
        at_sync = 0
        total_eligible = 0  # Only count ACTIVE (not MASKED) threads
        
        for warp in self.warps:
            # Check if warp is divergent (has entries on divergence stack)
            is_divergent = len(warp._divergence_stack) > 0
            
            for i, thread in enumerate(warp.threads):
                # Skip finished threads
                if thread.is_finished:
                    continue
                
                # Only count ACTIVE threads (not MASKED due to divergence)
                if warp._active_mask[i] and thread.status == ThreadStatus.ACTIVE:
                    total_eligible += 1
                    # Check if thread is at a SYNC instruction
                    if thread.program and thread.pc < len(thread.program):
                        from pygpu.core.instruction import OpCode
                        inst = thread.program[thread.pc]
                        if inst.opcode == OpCode.SYNC:
                            at_sync += 1
                elif thread.status == ThreadStatus.MASKED and is_divergent:
                    # Warn: SYNC in divergent code will cause issues
                    # For now, we skip masked threads entirely
                    pass
        
        if at_sync > 0:
            if total_eligible > 0 and at_sync == total_eligible:
                # All eligible (active, non-masked) threads at barrier - release them
                for warp in self.warps:
                    for i, thread in enumerate(warp.threads):
                        if (not thread.is_finished and 
                            warp._active_mask[i] and 
                            thread.status == ThreadStatus.ACTIVE and
                            thread.program):
                            thread.pc += 1
                return True
            else:
                # Some active threads still running - wait
                return True
        
        return False
    
    def get_warp_states(self) -> List[dict]:
        """Get state information for all warps (for visualization)."""
        states = []
        for warp in self.warps:
            state = {
                "warp_id": warp.warp_id,
                "pc": warp.pc,
                "finished": warp.is_finished,
                "thread_states": warp.get_thread_states(),
                "active_mask": warp.active_mask,
            }
            states.append(state)
        return states
    
    def get_thread_grid(self) -> List[List[str]]:
        """
        Get a 2D grid of thread states for visualization.
        
        Returns a list of rows, where each row is a warp.
        """
        grid = []
        for warp in self.warps:
            row = warp.get_thread_states()
            grid.append(row)
        return grid
    
    def __repr__(self) -> str:
        finished = sum(1 for w in self.warps if w.is_finished)
        return (
            f"Block(id={self.block_id}, warps={self.num_warps}, "
            f"threads={self.num_threads}, finished_warps={finished})"
        )
