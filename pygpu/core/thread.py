"""
Thread - The smallest execution unit in the GPU.

Each thread has its own program counter, registers, and execution status.
Threads are grouped into Warps that execute in lockstep.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional
import numpy as np

from pygpu.core.alu import ALU, Precision
from pygpu.core.instruction import Instruction, OpCode, Program


class ThreadStatus(Enum):
    """Execution status of a thread."""
    ACTIVE = auto()      # Currently executing
    WAITING = auto()     # Waiting for memory or synchronization
    MASKED = auto()      # Masked out due to branch divergence
    FINISHED = auto()    # Completed execution
    ERROR = auto()       # Thread encountered an error


class RegisterFile:
    """
    Per-thread register file.
    
    Contains:
    - R0-R31: General purpose registers (floating point)
    - P0-P7: Predicate registers (boolean, for branching)
    """
    
    NUM_GENERAL_REGS = 32
    NUM_PREDICATE_REGS = 8
    
    def __init__(self, precision: Precision = Precision.FLOAT32):
        self.precision = precision
        self._dtype = precision.dtype
        
        # General purpose registers
        self._general: Dict[str, np.number] = {
            f"R{i}": self._dtype(0) for i in range(self.NUM_GENERAL_REGS)
        }
        
        # Predicate registers (for conditional branching)
        self._predicates: Dict[str, bool] = {
            f"P{i}": False for i in range(self.NUM_PREDICATE_REGS)
        }
    
    def get(self, reg: str) -> Any:
        """Get the value of a register."""
        reg = reg.upper()
        if reg.startswith("P"):
            return self._predicates.get(reg, False)
        return self._general.get(reg, self._dtype(0))
    
    def set(self, reg: str, value: Any):
        """Set the value of a register."""
        reg = reg.upper()
        if reg.startswith("P"):
            self._predicates[reg] = bool(value)
        else:
            self._general[reg] = self._dtype(value)
    
    def __repr__(self) -> str:
        """Show non-zero registers."""
        non_zero = {k: v for k, v in self._general.items() if v != 0}
        preds = {k: v for k, v in self._predicates.items() if v}
        return f"RegisterFile(regs={non_zero}, preds={preds})"


class Thread:
    """
    A single GPU thread.
    
    Each thread has:
    - A unique thread ID within its block
    - A program counter (PC)
    - A register file
    - Execution status
    """
    
    def __init__(
        self,
        thread_id: int,
        block_id: int = 0,
        program: Optional[Program] = None,
        alu: Optional[ALU] = None,
        precision: Precision = Precision.FLOAT32,
    ):
        self.thread_id = thread_id
        self.block_id = block_id
        self.global_thread_id = thread_id  # Will be set by block
        
        self.pc: int = 0  # Program counter
        self.status = ThreadStatus.ACTIVE
        self.registers = RegisterFile(precision)
        self.alu = alu or ALU(precision)
        self.program = program
        
        # Memory access tracking (for visualization)
        self.last_memory_read: Optional[int] = None
        self.last_memory_write: Optional[int] = None
        self.cycles_waiting: int = 0
        
        # Memory latency simulation
        self._pending_load: Optional[dict] = None  # {dst_reg, address, cycles_remaining}
        
        # Execution history (for debugging/visualization)
        self.instruction_history: list[str] = []
        
        # Error information
        self.error_message: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        """Check if thread can execute."""
        return self.status == ThreadStatus.ACTIVE
    
    @property
    def is_finished(self) -> bool:
        """Check if thread has completed."""
        return self.status == ThreadStatus.FINISHED
    
    def fetch(self) -> Optional[Instruction]:
        """Fetch the next instruction."""
        if self.program is None or self.pc >= len(self.program):
            return None
        return self.program[self.pc]
    
    def execute(
        self,
        instruction: Instruction,
        global_memory: "GlobalMemory",
        shared_memory: Optional["SharedMemory"] = None,
    ) -> bool:
        """
        Execute a single instruction.
        
        Returns True if execution completed, False if stalled (waiting for memory).
        """
        from pygpu.core.memory import GlobalMemory, SharedMemory
        
        if self.status == ThreadStatus.MASKED:
            # Masked threads just advance PC
            self.pc += 1
            return True
        
        if self.status == ThreadStatus.ERROR:
            return False
        
        # Handle pending memory load (latency simulation)
        if self._pending_load is not None:
            self._pending_load["cycles_remaining"] -= 1
            if self._pending_load["cycles_remaining"] <= 0:
                # Load completed - write value to register
                try:
                    value = global_memory.read(self._pending_load["address"])
                    self.registers.set(self._pending_load["dst_reg"], value)
                except Exception as e:
                    self.status = ThreadStatus.ERROR
                    self.error_message = f"Memory read error: {e}"
                    return False
                self._pending_load = None
                self.status = ThreadStatus.ACTIVE
                self.pc += 1
                return True
            else:
                # Still waiting
                self.cycles_waiting += 1
                return False
        
        if self.status != ThreadStatus.ACTIVE:
            return False
        
        op = instruction.opcode
        
        # Record instruction for history
        self.instruction_history.append(str(instruction))
        
        # Clear memory tracking
        self.last_memory_read = None
        self.last_memory_write = None
        
        try:
            # Execute based on opcode
            if op == OpCode.NOP:
                pass
            
            elif op == OpCode.MOV:
                value = self._get_operand_value(instruction.src1)
                self.registers.set(instruction.dst, value)
            
            elif op == OpCode.THREAD_ID:
                self.registers.set(instruction.dst, self.thread_id)
            
            elif op == OpCode.BLOCK_ID:
                self.registers.set(instruction.dst, self.block_id)
            
            elif op == OpCode.LOAD:
                addr = self._get_address(instruction.src1)
                self.last_memory_read = addr
                # Initiate memory load with latency
                self._pending_load = {
                    "dst_reg": instruction.dst,
                    "address": addr,
                    "cycles_remaining": global_memory.LATENCY_CYCLES,
                }
                self.status = ThreadStatus.WAITING
                self.cycles_waiting = 1
                return False  # Stalled, don't advance PC yet
            
            elif op == OpCode.STORE:
                addr = self._get_address(instruction.src1)
                value = self.registers.get(instruction.dst)
                self.last_memory_write = addr
                global_memory.write(addr, value)
            
            elif op == OpCode.ADD:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.add(a, b))
            
            elif op == OpCode.SUB:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.sub(a, b))
            
            elif op == OpCode.MUL:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.mul(a, b))
            
            elif op == OpCode.DIV:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.div(a, b))
            
            elif op == OpCode.IDIV:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.idiv(a, b))
            
            elif op == OpCode.MOD:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.mod(a, b))
            
            elif op == OpCode.NEG:
                a = self._get_operand_value(instruction.src1)
                self.registers.set(instruction.dst, self.alu.neg(a))
            
            elif op == OpCode.ABS:
                a = self._get_operand_value(instruction.src1)
                self.registers.set(instruction.dst, self.alu.abs(a))
            
            elif op == OpCode.SQRT:
                a = self._get_operand_value(instruction.src1)
                self.registers.set(instruction.dst, self.alu.sqrt(a))
            
            elif op == OpCode.FMA:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                c = self._get_operand_value(instruction.src3)
                self.registers.set(instruction.dst, self.alu.fma(a, b, c))
            
            elif op == OpCode.CMP_EQ:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.eq(a, b))
            
            elif op == OpCode.CMP_NE:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.ne(a, b))
            
            elif op == OpCode.CMP_LT:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.lt(a, b))
            
            elif op == OpCode.CMP_LE:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.le(a, b))
            
            elif op == OpCode.CMP_GT:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.gt(a, b))
            
            elif op == OpCode.CMP_GE:
                a = self._get_operand_value(instruction.src1)
                b = self._get_operand_value(instruction.src2)
                self.registers.set(instruction.dst, self.alu.ge(a, b))
            
            elif op == OpCode.JMP:
                target = instruction.dst  # Label or address
                if isinstance(target, str) and self.program:
                    self.pc = self.program.get_label_address(target)
                else:
                    self.pc = int(target)
                return True  # Don't increment PC
            
            elif op == OpCode.BRA:
                predicate = self.registers.get(instruction.src1)
                if predicate:
                    target = instruction.dst
                    if isinstance(target, str) and self.program:
                        self.pc = self.program.get_label_address(target)
                    else:
                        self.pc = int(target)
                    return True
            
            elif op == OpCode.BRA_N:
                predicate = self.registers.get(instruction.src1)
                if not predicate:
                    target = instruction.dst
                    if isinstance(target, str) and self.program:
                        self.pc = self.program.get_label_address(target)
                    else:
                        self.pc = int(target)
                    return True
            
            elif op == OpCode.JOIN:
                # JOIN is a reconvergence point - handled at warp level
                # Thread just advances PC
                pass
            
            elif op == OpCode.RET:
                self.status = ThreadStatus.FINISHED
                return True
            
            elif op == OpCode.SYNC:
                # Synchronization is handled at the Warp/Block level
                pass
        
        except Exception as e:
            # Catch any execution errors and mark thread as errored
            self.status = ThreadStatus.ERROR
            self.error_message = f"Execution error at PC {self.pc}: {e}"
            return False
        
        # Advance program counter
        self.pc += 1
        
        # Check if we've reached the end
        if self.program and self.pc >= len(self.program):
            self.status = ThreadStatus.FINISHED
        
        return True
    
    def _get_operand_value(self, operand: Any) -> Any:
        """Get the value of an operand (register or immediate)."""
        if operand is None:
            return 0
        if isinstance(operand, str) and operand[0].upper() in ("R", "P"):
            return self.registers.get(operand)
        return operand
    
    def _get_address(self, operand: Any) -> int:
        """Get a memory address from an operand."""
        if isinstance(operand, tuple):
            op_type, value = operand
            if op_type == "mem_reg":
                return int(self.registers.get(value))
            elif op_type == "mem_imm":
                return value
        if isinstance(operand, str):
            return int(self.registers.get(operand))
        return int(operand)
    
    def __repr__(self) -> str:
        return (
            f"Thread(id={self.thread_id}, block={self.block_id}, "
            f"pc={self.pc}, status={self.status.name})"
        )
