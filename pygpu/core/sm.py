"""
Streaming Multiprocessor (SM) - The hardware unit that executes blocks.

Each SM can run one or more blocks concurrently. The SM contains the
execution units, register files, and shared memory for its blocks.
"""

from typing import List, Optional
from collections import deque

from pygpu.core.block import Block
from pygpu.core.instruction import Program
from pygpu.core.memory import GlobalMemory
from pygpu.core.alu import Precision


class StreamingMultiprocessor:
    """
    A Streaming Multiprocessor that executes thread blocks.
    
    In real GPUs, each SM has:
    - Multiple execution units (CUDA cores)
    - A warp scheduler
    - Register file banks
    - Shared memory
    
    Our SM limits the number of warps that can issue per cycle for realism.
    """
    
    MAX_BLOCKS_PER_SM = 4  # Maximum concurrent blocks
    MAX_WARPS_PER_SM = 16  # Maximum concurrent warps
    MAX_WARPS_PER_CYCLE = 2  # Maximum warps that can issue instructions per cycle
    
    def __init__(
        self,
        sm_id: int,
        precision: Precision = Precision.FLOAT32,
    ):
        self.sm_id = sm_id
        self.precision = precision
        
        # Currently executing blocks
        self.active_blocks: List[Block] = []
        
        # Queue of blocks waiting to be scheduled
        self.pending_blocks: deque[Block] = deque()
        
        # Round-robin warp scheduler state
        self._next_warp_index: int = 0
        
        # Execution tracking
        self.cycles_executed: int = 0
        self.blocks_completed: int = 0
    
    @property
    def is_idle(self) -> bool:
        """Check if the SM has no work to do."""
        return len(self.active_blocks) == 0 and len(self.pending_blocks) == 0
    
    @property
    def is_finished(self) -> bool:
        """Check if all assigned work is complete."""
        return self.is_idle
    
    @property
    def can_accept_block(self) -> bool:
        """Check if the SM can accept another block."""
        return len(self.active_blocks) < self.MAX_BLOCKS_PER_SM
    
    def submit_block(self, block: Block):
        """Submit a block for execution on this SM."""
        if self.can_accept_block:
            self.active_blocks.append(block)
        else:
            self.pending_blocks.append(block)
    
    def step(self, global_memory: GlobalMemory) -> bool:
        """
        Execute one cycle on active blocks with limited warp issue.
        
        Returns True if any block made progress.
        """
        # Try to schedule pending blocks
        self._schedule_pending_blocks()
        
        if not self.active_blocks:
            return False
        
        # Collect all ready warps from all active blocks
        ready_warps = []
        for block in self.active_blocks:
            for warp in block.warps:
                if not warp.is_finished and not warp.is_stalled:
                    ready_warps.append((block, warp))
        
        if not ready_warps:
            # Check if there are stalled warps that might make progress
            for block in self.active_blocks:
                for warp in block.warps:
                    if not warp.is_finished:
                        # Let stalled warps process their memory requests
                        warp.step(global_memory, block.shared_memory)
            return False
        
        # Round-robin select up to MAX_WARPS_PER_CYCLE warps to execute
        any_progress = False
        warps_issued = 0
        
        # Start from where we left off last cycle
        start_index = self._next_warp_index % len(ready_warps) if ready_warps else 0
        
        for i in range(len(ready_warps)):
            if warps_issued >= self.MAX_WARPS_PER_CYCLE:
                break
            
            idx = (start_index + i) % len(ready_warps)
            block, warp = ready_warps[idx]
            
            progress = warp.step(global_memory, block.shared_memory)
            if progress:
                any_progress = True
                warps_issued += 1
        
        # Update round-robin index for next cycle
        self._next_warp_index = (start_index + warps_issued) % max(1, len(ready_warps))
        
        # Check for finished blocks
        finished_blocks = []
        for block in self.active_blocks:
            if block.is_finished:
                finished_blocks.append(block)
            else:
                block.cycles_executed += 1
                block.shared_memory.clear_access_tracking()
        
        # Remove finished blocks
        for block in finished_blocks:
            self.active_blocks.remove(block)
            self.blocks_completed += 1
        
        self.cycles_executed += 1
        
        return any_progress
    
    def _schedule_pending_blocks(self):
        """Move pending blocks to active if there's capacity."""
        while self.pending_blocks and self.can_accept_block:
            block = self.pending_blocks.popleft()
            self.active_blocks.append(block)
    
    def get_utilization(self) -> float:
        """Get SM utilization as a percentage."""
        if not self.active_blocks:
            return 0.0
        
        total_warps = sum(len(b.warps) for b in self.active_blocks)
        return min(1.0, total_warps / self.MAX_WARPS_PER_SM)
    
    def get_state(self) -> dict:
        """Get current state for visualization."""
        return {
            "sm_id": self.sm_id,
            "active_blocks": len(self.active_blocks),
            "pending_blocks": len(self.pending_blocks),
            "utilization": self.get_utilization(),
            "cycles": self.cycles_executed,
            "blocks_completed": self.blocks_completed,
        }
    
    def __repr__(self) -> str:
        return (
            f"SM(id={self.sm_id}, active_blocks={len(self.active_blocks)}, "
            f"pending={len(self.pending_blocks)}, completed={self.blocks_completed})"
        )
