"""
Instruction Set Architecture (ISA) for the GPU simulator.

Defines the opcodes and instruction format that the GPU understands.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any, List


class OpCode(Enum):
    """GPU instruction opcodes."""
    
    # Data movement
    LOAD = auto()      # Load from memory to register
    STORE = auto()     # Store from register to memory
    MOV = auto()       # Move between registers or load immediate
    
    # Arithmetic
    ADD = auto()       # Add two registers
    SUB = auto()       # Subtract
    MUL = auto()       # Multiply
    DIV = auto()       # Divide (floating point)
    IDIV = auto()      # Integer divide (floor)
    MOD = auto()       # Modulo
    NEG = auto()       # Negate
    ABS = auto()       # Absolute value
    SQRT = auto()      # Square root
    FMA = auto()       # Fused multiply-add
    
    # Comparison (sets predicate register)
    CMP_EQ = auto()    # Compare equal
    CMP_NE = auto()    # Compare not equal
    CMP_LT = auto()    # Compare less than
    CMP_LE = auto()    # Compare less or equal
    CMP_GT = auto()    # Compare greater than
    CMP_GE = auto()    # Compare greater or equal
    
    # Control flow
    JMP = auto()       # Unconditional jump
    BRA = auto()       # Branch if predicate true
    BRA_N = auto()     # Branch if predicate false
    SYNC = auto()      # Synchronization barrier
    RET = auto()       # Return / end kernel
    JOIN = auto()      # Reconvergence point after divergent branch
    
    # Special
    NOP = auto()       # No operation
    THREAD_ID = auto() # Get thread ID into register
    BLOCK_ID = auto()  # Get block ID into register


class MemorySpace(Enum):
    """Memory space specifiers for load/store operations."""
    GLOBAL = auto()    # Global memory (slow, all threads)
    SHARED = auto()    # Shared memory (fast, block-local)
    LOCAL = auto()     # Local/private memory (per-thread)


@dataclass
class Instruction:
    """
    Represents a single GPU instruction.
    
    Format: OPCODE DST, SRC1 [, SRC2] [, SRC3]
    
    Registers are named R0-R31 for general purpose, P0-P7 for predicates.
    """
    
    opcode: OpCode
    dst: Optional[str] = None       # Destination register (e.g., "R0", "P0")
    src1: Optional[Any] = None      # Source 1 (register name or immediate value)
    src2: Optional[Any] = None      # Source 2 
    src3: Optional[Any] = None      # Source 3 (for FMA)
    memory_space: MemorySpace = MemorySpace.GLOBAL  # For LOAD/STORE
    label: Optional[str] = None     # Optional label for this instruction
    
    def __str__(self) -> str:
        """Human-readable assembly representation."""
        parts = [self.opcode.name]
        
        if self.dst:
            parts.append(self.dst)
        if self.src1 is not None:
            parts.append(str(self.src1))
        if self.src2 is not None:
            parts.append(str(self.src2))
        if self.src3 is not None:
            parts.append(str(self.src3))
            
        result = parts[0]
        if len(parts) > 1:
            result += " " + ", ".join(parts[1:])
            
        if self.label:
            result = f"{self.label}: {result}"
            
        return result
    
    @classmethod
    def parse(cls, line: str) -> "Instruction":
        """
        Parse an assembly line into an Instruction.
        
        Examples:
            "ADD R0, R1, R2"   -> Add R1 and R2, store in R0
            "LOAD R0, [100]"   -> Load from global memory address 100
            "MOV R0, 42"       -> Move immediate 42 into R0
        """
        line = line.strip()
        label = None
        
        # Strip inline comments
        if "#" in line:
            line = line.split("#")[0].strip()
        if "//" in line:
            line = line.split("//")[0].strip()
        
        # Check for label
        if ":" in line:
            label_part, line = line.split(":", 1)
            label = label_part.strip()
            line = line.strip()
        
        # Split into opcode and operands
        parts = line.replace(",", " ").split()
        if not parts:
            return cls(opcode=OpCode.NOP, label=label)
        
        opcode_str = parts[0].upper()
        try:
            opcode = OpCode[opcode_str]
        except KeyError:
            raise ValueError(f"Unknown opcode: {opcode_str}")
        
        operands = parts[1:] if len(parts) > 1 else []
        
        # Parse operands
        dst = operands[0] if len(operands) > 0 else None
        src1 = cls._parse_operand(operands[1]) if len(operands) > 1 else None
        src2 = cls._parse_operand(operands[2]) if len(operands) > 2 else None
        src3 = cls._parse_operand(operands[3]) if len(operands) > 3 else None
        
        return cls(
            opcode=opcode,
            dst=dst,
            src1=src1,
            src2=src2,
            src3=src3,
            label=label,
        )
    
    @staticmethod
    def _parse_operand(op: str) -> Any:
        """Parse a single operand (register, immediate, or memory address)."""
        op = op.strip()
        
        # Memory address: [addr] or [Rn]
        if op.startswith("[") and op.endswith("]"):
            inner = op[1:-1]
            if inner.startswith("R") or inner.startswith("r"):
                return ("mem_reg", inner.upper())
            else:
                return ("mem_imm", int(inner))
        
        # Register
        if op[0].upper() in ("R", "P"):
            return op.upper()
        
        # Immediate value
        try:
            if "." in op:
                return float(op)
            return int(op)
        except ValueError:
            return op  # Label reference


class Program:
    """
    A collection of instructions forming a GPU kernel.
    """
    
    def __init__(self, instructions: Optional[List[Instruction]] = None):
        self.instructions: List[Instruction] = instructions or []
        self._labels: dict[str, int] = {}
        self._build_label_index()
    
    def _build_label_index(self):
        """Build an index of labels to instruction indices."""
        self._labels.clear()
        for i, inst in enumerate(self.instructions):
            if inst.label:
                self._labels[inst.label] = i
    
    def add(self, instruction: Instruction):
        """Add an instruction to the program."""
        if instruction.label:
            self._labels[instruction.label] = len(self.instructions)
        self.instructions.append(instruction)
    
    def get_label_address(self, label: str) -> int:
        """Get the instruction index for a label."""
        return self._labels.get(label, -1)
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]
    
    @classmethod
    def from_assembly(cls, code: str) -> "Program":
        """Parse assembly code into a Program."""
        program = cls()
        for line in code.strip().split("\n"):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            program.add(Instruction.parse(line))
        return program
    
    def __str__(self) -> str:
        return "\n".join(str(inst) for inst in self.instructions)
