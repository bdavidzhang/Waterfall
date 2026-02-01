"""
Warp - A group of threads that execute in lockstep (SIMT).

The Warp is the fundamental execution unit in SIMT architecture.
All threads in a warp execute the same instruction at the same time,
but may operate on different data (SIMD-like behavior).

Branch divergence is handled by masking threads that take different paths.
"""

from typing import List, Optional, Set
from dataclasses import dataclass, field

from pygpu.core.thread import Thread, ThreadStatus
from pygpu.core.instruction import Instruction, OpCode, Program
from pygpu.core.memory import GlobalMemory, SharedMemory
from pygpu.core.alu import Precision


@dataclass
class DivergenceStackEntry:
    """Entry for tracking branch divergence."""
    reconvergence_pc: int          # PC where threads reconverge (JOIN instruction)
    active_mask: List[bool]        # Which threads were active before divergence
    taken_mask: List[bool]         # Which threads took the branch
    executing_taken: bool = True   # True if currently executing taken path


class Warp:
    """
    A warp of threads executing in lockstep.
    
    In NVIDIA GPUs, a warp is typically 32 threads.
    For educational purposes, we default to 8 threads for easier visualization.
    """
    
    DEFAULT_SIZE = 8  # Threads per warp
    
    def __init__(
        self,
        warp_id: int,
        block_id: int,
        num_threads: int = DEFAULT_SIZE,
        program: Optional[Program] = None,
        precision: Precision = Precision.FLOAT32,
        thread_id_offset: int = 0,
    ):
        self.warp_id = warp_id
        self.block_id = block_id
        self.num_threads = num_threads
        self.program = program
        self.precision = precision
        
        # Create threads
        self.threads: List[Thread] = []
        for i in range(num_threads):
            thread = Thread(
                thread_id=thread_id_offset + i,
                block_id=block_id,
                program=program,
                precision=precision,
            )
            self.threads.append(thread)
        
        # Active mask - which threads are currently executing
        # False = masked out (divergent or finished)
        self._active_mask: List[bool] = [True] * num_threads
        
        # Divergence stack for handling nested branches
        self._divergence_stack: List[DivergenceStackEntry] = []
        
        # Synchronization barrier tracking
        self._at_barrier: Set[int] = set()
        
        # Execution tracking
        self.instructions_executed: int = 0
        self.cycles_stalled: int = 0
    
    @property
    def active_mask(self) -> List[bool]:
        """Get the current active mask."""
        return self._active_mask.copy()
    
    @property
    def pc(self) -> int:
        """Get the current program counter (same for all active threads)."""
        for i, active in enumerate(self._active_mask):
            if active:
                return self.threads[i].pc
        # All threads finished, return last PC
        return self.threads[0].pc
    
    @property
    def is_finished(self) -> bool:
        """Check if all threads in the warp have finished."""
        return all(t.is_finished for t in self.threads)
    
    @property
    def is_stalled(self) -> bool:
        """Check if the warp is stalled (e.g., waiting for memory)."""
        return any(t.status == ThreadStatus.WAITING for t in self.threads)
    
    def step(
        self,
        global_memory: GlobalMemory,
        shared_memory: Optional[SharedMemory] = None,
    ) -> bool:
        """
        Execute one instruction across all active threads.
        
        Returns True if the warp made progress, False if stalled or finished.
        """
        if self.is_finished:
            return False
        
        # Check if any threads are waiting for memory (latency simulation)
        any_waiting = any(
            self._active_mask[i] and self.threads[i].status == ThreadStatus.WAITING
            for i in range(self.num_threads)
        )
        if any_waiting:
            # Process waiting threads
            any_progress = False
            for i, thread in enumerate(self.threads):
                if self._active_mask[i] and thread.status == ThreadStatus.WAITING:
                    # Call execute to process pending memory operations
                    if thread._pending_load is not None:
                        progress = thread.execute(None, global_memory, shared_memory)
                        if progress:
                            any_progress = True
            if any_progress:
                self._update_active_mask()
                return True
            self.cycles_stalled += 1
            return False
        
        # Get current instruction from the PC
        current_pc = self.pc
        if self.program is None or current_pc >= len(self.program):
            # No more instructions, mark all threads as finished
            for thread in self.threads:
                if thread.status == ThreadStatus.ACTIVE:
                    thread.status = ThreadStatus.FINISHED
            return False
        
        instruction = self.program[current_pc]
        
        # Handle synchronization barriers
        if instruction.opcode == OpCode.SYNC:
            return self._handle_sync()
        
        # Handle JOIN instruction (reconvergence point)
        if instruction.opcode == OpCode.JOIN:
            return self._handle_join()
        
        # Handle branch instructions (divergence)
        if instruction.opcode in (OpCode.BRA, OpCode.BRA_N):
            return self._handle_branch(instruction, global_memory, shared_memory)
        
        # Execute instruction on all active threads
        all_stalled = True
        for i, thread in enumerate(self.threads):
            if self._active_mask[i] and not thread.is_finished:
                result = thread.execute(instruction, global_memory, shared_memory)
                if result:
                    all_stalled = False
        
        if all_stalled and any(self._active_mask):
            # All active threads stalled (e.g., memory latency)
            self.cycles_stalled += 1
            return False
        
        self.instructions_executed += 1
        
        # Check for reconvergence after instruction
        self._check_reconvergence()
        
        # Update active mask based on thread status
        self._update_active_mask()
        
        return True
    
    def _handle_branch(
        self,
        instruction: Instruction,
        global_memory: GlobalMemory,
        shared_memory: Optional[SharedMemory],
    ) -> bool:
        """
        Handle a conditional branch instruction.
        
        This is where branch divergence occurs. We:
        1. Evaluate the condition for each active thread
        2. If threads diverge, push onto the divergence stack
        3. Execute the taken path first, then the not-taken path
        4. Threads reconverge at the JOIN instruction
        """
        is_bra_n = instruction.opcode == OpCode.BRA_N
        predicate_reg = instruction.src1
        target = instruction.dst
        
        # Get target PC
        if isinstance(target, str) and self.program:
            target_pc = self.program.get_label_address(target)
        else:
            target_pc = int(target)
        
        # Evaluate condition for each thread
        taken_mask = []
        for i, thread in enumerate(self.threads):
            if self._active_mask[i]:
                predicate = thread.registers.get(predicate_reg)
                if is_bra_n:
                    predicate = not predicate
                taken_mask.append(predicate)
            else:
                taken_mask.append(False)
        
        # Count how many threads take/don't take the branch
        num_taken = sum(taken_mask)
        num_not_taken = sum(self._active_mask) - num_taken
        
        if num_taken == 0:
            # All threads don't take the branch - just advance PC
            for thread in self.threads:
                if thread.status == ThreadStatus.ACTIVE:
                    thread.pc += 1
            return True
        
        if num_not_taken == 0:
            # All threads take the branch - jump to target
            for thread in self.threads:
                if thread.status == ThreadStatus.ACTIVE:
                    thread.pc = target_pc
            return True
        
        # DIVERGENCE! Some threads take, some don't
        # We execute the TAKEN path first, then NOT-TAKEN path
        # Find the reconvergence point (JOIN instruction)
        reconvergence_pc = self._find_reconvergence_point(self.pc)
        
        # Push divergence onto stack
        entry = DivergenceStackEntry(
            reconvergence_pc=reconvergence_pc,
            active_mask=self._active_mask.copy(),
            taken_mask=taken_mask.copy(),
            executing_taken=True,
        )
        self._divergence_stack.append(entry)
        
        # Execute threads that TAKE the branch first
        for i, (active, taken) in enumerate(zip(self._active_mask, taken_mask)):
            if active:
                if taken:
                    self.threads[i].pc = target_pc
                    self.threads[i].status = ThreadStatus.ACTIVE
                else:
                    # Mask not-taken threads, but set their PC for when they resume
                    self.threads[i].pc = self.pc + 1  # Fall-through path
                    self.threads[i].status = ThreadStatus.MASKED
                    self._active_mask[i] = False
        
        return True
    
    def _find_reconvergence_point(self, branch_pc: int) -> int:
        """
        Find the JOIN instruction that corresponds to a branch.
        
        This is a simplified approach - looks for the next JOIN instruction.
        A more sophisticated approach would track nesting levels.
        """
        if self.program is None:
            return branch_pc + 1
        
        # Look for JOIN instruction after the branch
        for pc in range(branch_pc + 1, len(self.program)):
            if self.program[pc].opcode == OpCode.JOIN:
                return pc
        
        # No JOIN found - reconverge at end of program
        return len(self.program)
    
    def _handle_join(self) -> bool:
        """
        Handle JOIN instruction - the reconvergence point after divergence.
        
        This is where masked threads are restored and all threads continue together.
        """
        if not self._divergence_stack:
            # No divergence to reconverge - just advance PC
            for thread in self.threads:
                if thread.status == ThreadStatus.ACTIVE:
                    thread.pc += 1
            return True
        
        entry = self._divergence_stack[-1]
        
        # Check if we're at the correct reconvergence point
        if self.pc == entry.reconvergence_pc:
            if entry.executing_taken:
                # Finished TAKEN path, now execute NOT-TAKEN path
                entry.executing_taken = False
                
                # Restore NOT-TAKEN threads
                for i in range(self.num_threads):
                    if entry.active_mask[i] and not entry.taken_mask[i]:
                        if not self.threads[i].is_finished:
                            self.threads[i].status = ThreadStatus.ACTIVE
                            self._active_mask[i] = True
                    elif entry.taken_mask[i]:
                        # Mask TAKEN threads now
                        self.threads[i].status = ThreadStatus.MASKED
                        self._active_mask[i] = False
                
                # Check if there are any NOT-TAKEN threads to execute
                if not any(self._active_mask):
                    # All threads finished or no not-taken threads
                    self._divergence_stack.pop()
                    self._restore_all_threads(entry)
                    for thread in self.threads:
                        if thread.status == ThreadStatus.ACTIVE:
                            thread.pc += 1
                    return True
                
                # NOT-TAKEN threads continue from their saved PC (fall-through)
                # They should already have their PC set correctly
                return True
            else:
                # Finished NOT-TAKEN path, reconverge all threads
                self._divergence_stack.pop()
                self._restore_all_threads(entry)
                
                # All threads advance past JOIN
                for thread in self.threads:
                    if thread.status == ThreadStatus.ACTIVE:
                        thread.pc += 1
                return True
        
        # Not at reconvergence point yet - just advance
        for thread in self.threads:
            if thread.status == ThreadStatus.ACTIVE:
                thread.pc += 1
        return True
    
    def _restore_all_threads(self, entry: DivergenceStackEntry):
        """Restore all threads that were active before divergence."""
        for i in range(self.num_threads):
            if entry.active_mask[i] and not self.threads[i].is_finished:
                self.threads[i].status = ThreadStatus.ACTIVE
                self._active_mask[i] = True
    
    def _check_reconvergence(self):
        """Check if we need to restore masked threads (legacy - now handled by JOIN)."""
        # With explicit JOIN instructions, reconvergence is handled in _handle_join
        # This method is kept for backward compatibility with programs without JOIN
        if not self._divergence_stack:
            return
        
        entry = self._divergence_stack[-1]
        
        # Check if all currently active threads have reached the reconvergence PC
        # This only triggers if there's no explicit JOIN instruction
        all_at_reconvergence = all(
            not self._active_mask[i] or 
            self.threads[i].is_finished or
            self.threads[i].pc >= entry.reconvergence_pc
            for i in range(self.num_threads)
        )
        
        if all_at_reconvergence:
            if entry.executing_taken:
                # Switch to NOT-TAKEN path
                entry.executing_taken = False
                
                not_taken_mask = [
                    entry.active_mask[i] and not entry.taken_mask[i]
                    for i in range(self.num_threads)
                ]
                
                if any(not_taken_mask):
                    for i, should_run in enumerate(not_taken_mask):
                        if should_run and not self.threads[i].is_finished:
                            self.threads[i].status = ThreadStatus.ACTIVE
                            self._active_mask[i] = True
                    # Mask taken threads
                    for i in range(self.num_threads):
                        if entry.taken_mask[i]:
                            self._active_mask[i] = False
                else:
                    # No NOT-TAKEN threads, just pop
                    self._divergence_stack.pop()
                    self._restore_all_threads(entry)
            else:
                # Finished both paths
                self._divergence_stack.pop()
                self._restore_all_threads(entry)
    
    def _handle_sync(self) -> bool:
        """
        Handle synchronization barrier.
        
        All threads in the warp must reach the barrier before proceeding.
        """
        # In a warp, all threads execute in lockstep, so sync is automatic
        # The sync instruction is more relevant at the block level
        for thread in self.threads:
            if thread.status == ThreadStatus.ACTIVE:
                thread.pc += 1
        return True
    
    def _update_active_mask(self):
        """Update the active mask based on thread status."""
        for i, thread in enumerate(self.threads):
            if thread.status == ThreadStatus.FINISHED:
                self._active_mask[i] = False
            elif thread.status == ThreadStatus.MASKED:
                self._active_mask[i] = False
            elif thread.status == ThreadStatus.ACTIVE:
                # Keep current mask value (might be masked due to divergence)
                pass
    
    def get_thread_states(self) -> List[str]:
        """Get a list of thread states for visualization."""
        states = []
        for i, thread in enumerate(self.threads):
            if thread.is_finished:
                states.append("FINISHED")
            elif not self._active_mask[i]:
                states.append("MASKED")
            elif thread.status == ThreadStatus.WAITING:
                states.append("WAITING")
            else:
                states.append("ACTIVE")
        return states
    
    def __repr__(self) -> str:
        active = sum(self._active_mask)
        finished = sum(1 for t in self.threads if t.is_finished)
        return (
            f"Warp(id={self.warp_id}, block={self.block_id}, "
            f"pc={self.pc}, active={active}/{self.num_threads}, "
            f"finished={finished})"
        )
