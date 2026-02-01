"""
GPU - The top-level GPU simulator.

This is the main entry point for running GPU simulations.
It manages the memory, SMs, and provides the tick-based execution loop.
"""

from typing import Callable, Optional, List, Any, Union
import inspect
import ast

from pygpu.core.alu import Precision
from pygpu.core.memory import GlobalMemory
from pygpu.core.sm import StreamingMultiprocessor
from pygpu.core.block import Block
from pygpu.core.instruction import Program, Instruction, OpCode


class GPU:
    """
    The main GPU simulator class.
    
    Provides a cycle-accurate simulation of GPU execution with:
    - Configurable precision (16/32/64-bit)
    - Multiple Streaming Multiprocessors
    - Global memory
    - Tick-based execution for visualization
    """
    
    DEFAULT_NUM_SMS = 2
    
    def __init__(
        self,
        precision: str = "float32",
        num_sms: int = DEFAULT_NUM_SMS,
        global_memory_size: int = GlobalMemory.DEFAULT_SIZE,
    ):
        # Parse precision
        precision_map = {
            "float16": Precision.FLOAT16,
            "float32": Precision.FLOAT32,
            "float64": Precision.FLOAT64,
            "16": Precision.FLOAT16,
            "32": Precision.FLOAT32,
            "64": Precision.FLOAT64,
        }
        self.precision = precision_map.get(precision, Precision.FLOAT32)
        
        # Initialize global memory
        self.global_memory = GlobalMemory(global_memory_size, self.precision)
        
        # Initialize streaming multiprocessors
        self.sms: List[StreamingMultiprocessor] = [
            StreamingMultiprocessor(i, self.precision) for i in range(num_sms)
        ]
        
        # Execution state
        self.clock_cycle: int = 0
        self.program: Optional[Program] = None
        self.blocks: List[Block] = []
        
        # Visualization callback
        self._render_callback: Optional[Callable] = None
        self._step_delay: float = 0.0
        
        # Execution history for debugging
        self.cycle_history: List[dict] = []
    
    @property
    def is_busy(self) -> bool:
        """Check if any SM is still executing."""
        return any(not sm.is_finished for sm in self.sms)
    
    def load_program(self, kernel: Union[Callable, Program, str]):
        """
        Load a program (kernel) for execution.
        
        Accepts:
        - A Python function (will be compiled to our ISA)
        - A Program object
        - Assembly code as a string
        """
        if isinstance(kernel, Program):
            self.program = kernel
        elif isinstance(kernel, str):
            self.program = Program.from_assembly(kernel)
        elif callable(kernel):
            self.program = self._compile_kernel(kernel)
        else:
            raise ValueError(f"Unknown kernel type: {type(kernel)}")
    
    def _compile_kernel(self, kernel: Callable) -> Program:
        """
        Compile a Python kernel function to our assembly ISA.
        
        This is a simplified compiler that handles basic operations.
        For full Python support, a more sophisticated compiler would be needed.
        """
        source = inspect.getsource(kernel)
        tree = ast.parse(source)
        
        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
        
        if func_def is None:
            raise ValueError("Could not find function definition")
        
        compiler = KernelCompiler()
        return compiler.compile(func_def)
    
    def launch(
        self,
        blocks: int = 1,
        threads_per_block: int = 8,
        threads_per_warp: int = 8,
    ):
        """
        Launch the loaded program with the specified grid configuration.
        
        Args:
            blocks: Number of thread blocks
            threads_per_block: Number of threads in each block
            threads_per_warp: Number of threads per warp (default: 8 for visualization)
        """
        if self.program is None:
            raise ValueError("No program loaded. Call load_program() first.")
        
        self.clock_cycle = 0
        self.blocks.clear()
        self.cycle_history.clear()
        
        # Reset SMs
        for sm in self.sms:
            sm.active_blocks.clear()
            sm.pending_blocks.clear()
            sm.cycles_executed = 0
            sm.blocks_completed = 0
        
        # Create blocks and distribute to SMs
        for block_id in range(blocks):
            block = Block(
                block_id=block_id,
                num_threads=threads_per_block,
                threads_per_warp=threads_per_warp,
                program=self.program,
                precision=self.precision,
            )
            self.blocks.append(block)
            
            # Round-robin distribution to SMs
            sm_id = block_id % len(self.sms)
            self.sms[sm_id].submit_block(block)
    
    def tick(self) -> bool:
        """
        Simulate one clock cycle.
        
        Returns True if any work was done, False if all finished.
        """
        self.clock_cycle += 1
        
        # Record pre-cycle state for history
        pre_state = self.get_state()
        
        # Clear memory access tracking
        self.global_memory.clear_access_tracking()
        
        # Step each SM
        any_progress = False
        for sm in self.sms:
            if not sm.is_finished:
                progress = sm.step(self.global_memory)
                if progress:
                    any_progress = True
        
        # Record memory access history
        self.global_memory.record_cycle(self.clock_cycle)
        
        # Store cycle history
        self.cycle_history.append({
            "cycle": self.clock_cycle,
            "state": pre_state,
            "memory_reads": list(self.global_memory._read_addresses),
            "memory_writes": list(self.global_memory._write_addresses),
        })
        
        return any_progress
    
    def run(
        self,
        max_cycles: int = 10000,
        render_callback: Optional[Callable] = None,
    ) -> int:
        """
        Run the simulation until completion.
        
        Args:
            max_cycles: Maximum number of cycles before stopping
            render_callback: Optional callback for visualization (called each cycle)
        
        Returns:
            Number of cycles executed
        """
        self._render_callback = render_callback
        
        while self.is_busy and self.clock_cycle < max_cycles:
            self.tick()
            
            if self._render_callback:
                self._render_callback(self)
        
        return self.clock_cycle
    
    def step(self) -> dict:
        """
        Execute a single step and return state (for interactive mode).
        """
        self.tick()
        return self.get_state()
    
    def get_state(self) -> dict:
        """Get the current state of the GPU for visualization."""
        return {
            "cycle": self.clock_cycle,
            "sms": [sm.get_state() for sm in self.sms],
            "blocks": [
                {
                    "block_id": b.block_id,
                    "thread_grid": b.get_thread_grid(),
                    "finished": b.is_finished,
                }
                for b in self.blocks
            ],
            "memory_reads": list(self.global_memory._read_addresses),
            "memory_writes": list(self.global_memory._write_addresses),
            "finished": not self.is_busy,
        }
    
    def get_thread_grid(self) -> List[List[List[str]]]:
        """
        Get the thread state grid for all blocks.
        
        Returns: List of blocks, each containing a 2D grid of thread states.
        """
        return [block.get_thread_grid() for block in self.blocks]
    
    def get_memory_heatmap(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> List[int]:
        """Get the memory access heatmap."""
        return self.global_memory.get_heatmap(start, end)
    
    def __repr__(self) -> str:
        return (
            f"GPU(precision={self.precision.value}, sms={len(self.sms)}, "
            f"cycle={self.clock_cycle}, busy={self.is_busy})"
        )


class KernelCompiler:
    """
    A simple compiler that converts Python kernel functions to our ISA.
    
    This handles a subset of Python:
    - Variable assignments
    - Arithmetic operations (+, -, *, /)
    - Array indexing (global_mem[idx])
    - Simple conditionals (if/else)
    """
    
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.variables: dict[str, str] = {}  # var_name -> register
        self.next_reg: int = 0
        self.label_counter: int = 0
    
    def _alloc_reg(self) -> str:
        """Allocate a new general-purpose register."""
        reg = f"R{self.next_reg}"
        self.next_reg += 1
        return reg
    
    def _new_label(self, prefix: str = "L") -> str:
        """Create a new unique label."""
        label = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return label
    
    def compile(self, func_def: ast.FunctionDef) -> Program:
        """Compile a function definition to a Program."""
        self.instructions = []
        self.variables = {}
        self.next_reg = 0
        self.label_counter = 0
        
        # Handle function arguments - map to special registers/values
        for i, arg in enumerate(func_def.args.args):
            arg_name = arg.arg
            if arg_name == "thread_id":
                # Thread ID is fetched via THREAD_ID instruction
                reg = self._alloc_reg()
                self.variables[arg_name] = reg
                self.instructions.append(Instruction(
                    opcode=OpCode.THREAD_ID,
                    dst=reg,
                ))
            elif arg_name == "block_id":
                reg = self._alloc_reg()
                self.variables[arg_name] = reg
                self.instructions.append(Instruction(
                    opcode=OpCode.BLOCK_ID,
                    dst=reg,
                ))
            elif arg_name in ("global_mem", "shared_mem"):
                # Memory is accessed via LOAD/STORE, not a register
                pass
        
        # Compile function body
        for stmt in func_def.body:
            self._compile_stmt(stmt)
        
        # Add RET at end
        self.instructions.append(Instruction(opcode=OpCode.RET))
        
        return Program(self.instructions)
    
    def _compile_stmt(self, stmt: ast.stmt):
        """Compile a statement."""
        if isinstance(stmt, ast.Assign):
            self._compile_assign(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._compile_aug_assign(stmt)
        elif isinstance(stmt, ast.If):
            self._compile_if(stmt)
        elif isinstance(stmt, ast.For):
            self._compile_for(stmt)
        elif isinstance(stmt, ast.While):
            self._compile_while(stmt)
        elif isinstance(stmt, ast.Expr):
            # Expression statement (like a function call)
            self._compile_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            self.instructions.append(Instruction(opcode=OpCode.RET))
        elif isinstance(stmt, ast.Pass):
            self.instructions.append(Instruction(opcode=OpCode.NOP))
    
    def _compile_assign(self, stmt: ast.Assign):
        """Compile an assignment statement."""
        if len(stmt.targets) != 1:
            raise ValueError("Multiple assignment targets not supported")
        
        target = stmt.targets[0]
        
        if isinstance(target, ast.Subscript):
            # Store to memory: global_mem[idx] = value
            self._compile_store(target, stmt.value)
        elif isinstance(target, ast.Name):
            # Variable assignment
            value_reg = self._compile_expr(stmt.value)
            var_name = target.id
            
            if var_name in self.variables:
                # Move to existing register
                dst_reg = self.variables[var_name]
                if dst_reg != value_reg:
                    self.instructions.append(Instruction(
                        opcode=OpCode.MOV,
                        dst=dst_reg,
                        src1=value_reg,
                    ))
            else:
                # Allocate new register (or reuse the computed one)
                self.variables[var_name] = value_reg
    
    def _compile_aug_assign(self, stmt: ast.AugAssign):
        """Compile augmented assignment (+=, -=, etc.)."""
        if not isinstance(stmt.target, ast.Name):
            raise ValueError("Only simple augmented assignments supported")
        
        var_name = stmt.target.id
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not defined")
        
        var_reg = self.variables[var_name]
        value_reg = self._compile_expr(stmt.value)
        
        op_map = {
            ast.Add: OpCode.ADD,
            ast.Sub: OpCode.SUB,
            ast.Mult: OpCode.MUL,
            ast.Div: OpCode.DIV,
            ast.Mod: OpCode.MOD,
        }
        
        op_type = type(stmt.op)
        if op_type not in op_map:
            raise ValueError(f"Unsupported augmented operator: {op_type}")
        
        self.instructions.append(Instruction(
            opcode=op_map[op_type],
            dst=var_reg,
            src1=var_reg,
            src2=value_reg,
        ))
    
    def _compile_store(self, target: ast.Subscript, value: ast.expr):
        """Compile a store to memory."""
        # Get the address
        addr_reg = self._compile_expr(target.slice)
        
        # Get the value
        value_reg = self._compile_expr(value)
        
        self.instructions.append(Instruction(
            opcode=OpCode.STORE,
            dst=value_reg,
            src1=("mem_reg", addr_reg),
        ))
    
    def _compile_expr(self, expr: ast.expr) -> str:
        """
        Compile an expression and return the register containing the result.
        """
        if isinstance(expr, ast.Constant):
            # Immediate value
            reg = self._alloc_reg()
            self.instructions.append(Instruction(
                opcode=OpCode.MOV,
                dst=reg,
                src1=expr.value,
            ))
            return reg
        
        elif isinstance(expr, ast.Name):
            var_name = expr.id
            if var_name in self.variables:
                return self.variables[var_name]
            else:
                raise ValueError(f"Undefined variable: {var_name}")
        
        elif isinstance(expr, ast.BinOp):
            return self._compile_binop(expr)
        
        elif isinstance(expr, ast.UnaryOp):
            return self._compile_unaryop(expr)
        
        elif isinstance(expr, ast.Subscript):
            return self._compile_load(expr)
        
        elif isinstance(expr, ast.Compare):
            return self._compile_compare(expr)
        
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")
    
    def _compile_binop(self, expr: ast.BinOp) -> str:
        """Compile a binary operation."""
        left_reg = self._compile_expr(expr.left)
        right_reg = self._compile_expr(expr.right)
        result_reg = self._alloc_reg()
        
        op_map = {
            ast.Add: OpCode.ADD,
            ast.Sub: OpCode.SUB,
            ast.Mult: OpCode.MUL,
            ast.Div: OpCode.DIV,
            ast.Mod: OpCode.MOD,
        }
        
        op_type = type(expr.op)
        if op_type not in op_map:
            raise ValueError(f"Unsupported binary operator: {op_type}")
        
        self.instructions.append(Instruction(
            opcode=op_map[op_type],
            dst=result_reg,
            src1=left_reg,
            src2=right_reg,
        ))
        
        return result_reg
    
    def _compile_unaryop(self, expr: ast.UnaryOp) -> str:
        """Compile a unary operation."""
        operand_reg = self._compile_expr(expr.operand)
        result_reg = self._alloc_reg()
        
        if isinstance(expr.op, ast.USub):
            self.instructions.append(Instruction(
                opcode=OpCode.NEG,
                dst=result_reg,
                src1=operand_reg,
            ))
        else:
            raise ValueError(f"Unsupported unary operator: {type(expr.op)}")
        
        return result_reg
    
    def _compile_load(self, expr: ast.Subscript) -> str:
        """Compile a load from memory."""
        addr_reg = self._compile_expr(expr.slice)
        result_reg = self._alloc_reg()
        
        self.instructions.append(Instruction(
            opcode=OpCode.LOAD,
            dst=result_reg,
            src1=("mem_reg", addr_reg),
        ))
        
        return result_reg
    
    def _compile_compare(self, expr: ast.Compare) -> str:
        """Compile a comparison expression."""
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            raise ValueError("Only simple comparisons supported")
        
        left_reg = self._compile_expr(expr.left)
        right_reg = self._compile_expr(expr.comparators[0])
        result_reg = f"P{self.label_counter % 8}"  # Use predicate register
        self.label_counter += 1
        
        op_map = {
            ast.Eq: OpCode.CMP_EQ,
            ast.NotEq: OpCode.CMP_NE,
            ast.Lt: OpCode.CMP_LT,
            ast.LtE: OpCode.CMP_LE,
            ast.Gt: OpCode.CMP_GT,
            ast.GtE: OpCode.CMP_GE,
        }
        
        op_type = type(expr.ops[0])
        if op_type not in op_map:
            raise ValueError(f"Unsupported comparison: {op_type}")
        
        self.instructions.append(Instruction(
            opcode=op_map[op_type],
            dst=result_reg,
            src1=left_reg,
            src2=right_reg,
        ))
        
        return result_reg
    
    def _compile_if(self, stmt: ast.If):
        """Compile an if statement."""
        # Compile condition
        cond_reg = self._compile_expr(stmt.test)
        
        else_label = self._new_label("else")
        end_label = self._new_label("endif")
        
        # Branch to else if condition is false
        self.instructions.append(Instruction(
            opcode=OpCode.BRA_N,
            dst=else_label,
            src1=cond_reg,
        ))
        
        # Compile then body
        for s in stmt.body:
            self._compile_stmt(s)
        
        # Jump to end (skip else)
        if stmt.orelse:
            self.instructions.append(Instruction(
                opcode=OpCode.JMP,
                dst=end_label,
            ))
        
        # Else label
        self.instructions.append(Instruction(
            opcode=OpCode.NOP,
            label=else_label,
        ))
        
        # Compile else body
        for s in stmt.orelse:
            self._compile_stmt(s)
        
        # End label / JOIN point for reconvergence
        self.instructions.append(Instruction(
            opcode=OpCode.JOIN,
            label=end_label,
        ))
    
    def _compile_for(self, stmt: ast.For):
        """
        Compile a for loop.
        
        Supports: for i in range(start, end) or for i in range(end)
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ValueError("Only 'for i in range(...)' loops are supported")
        
        func = stmt.iter.func
        if not (isinstance(func, ast.Name) and func.id == "range"):
            raise ValueError("Only 'for i in range(...)' loops are supported")
        
        args = stmt.iter.args
        
        # Parse range arguments
        if len(args) == 1:
            start_val = 0
            end_expr = args[0]
            step_val = 1
        elif len(args) == 2:
            start_expr = args[0]
            end_expr = args[1]
            step_val = 1
            if isinstance(start_expr, ast.Constant):
                start_val = start_expr.value
            else:
                raise ValueError("Range start must be a constant")
        elif len(args) == 3:
            start_expr = args[0]
            end_expr = args[1]
            step_expr = args[2]
            if isinstance(start_expr, ast.Constant):
                start_val = start_expr.value
            else:
                raise ValueError("Range start must be a constant")
            if isinstance(step_expr, ast.Constant):
                step_val = step_expr.value
            else:
                raise ValueError("Range step must be a constant")
        else:
            raise ValueError("range() takes 1-3 arguments")
        
        # Get loop variable name
        if not isinstance(stmt.target, ast.Name):
            raise ValueError("Loop variable must be a simple name")
        loop_var = stmt.target.id
        
        # Allocate register for loop variable
        loop_reg = self._alloc_reg()
        self.variables[loop_var] = loop_reg
        
        # Initialize loop variable
        self.instructions.append(Instruction(
            opcode=OpCode.MOV,
            dst=loop_reg,
            src1=start_val,
        ))
        
        # Get end value into a register
        end_reg = self._compile_expr(end_expr)
        
        # Labels
        loop_start = self._new_label("loop_start")
        loop_end = self._new_label("loop_end")
        
        # Loop start label
        self.instructions.append(Instruction(
            opcode=OpCode.NOP,
            label=loop_start,
        ))
        
        # Check condition: loop_var < end
        cond_reg = f"P{self.label_counter % 8}"
        self.label_counter += 1
        self.instructions.append(Instruction(
            opcode=OpCode.CMP_LT,
            dst=cond_reg,
            src1=loop_reg,
            src2=end_reg,
        ))
        
        # Branch to end if condition is false
        self.instructions.append(Instruction(
            opcode=OpCode.BRA_N,
            dst=loop_end,
            src1=cond_reg,
        ))
        
        # Compile loop body
        for s in stmt.body:
            self._compile_stmt(s)
        
        # Increment loop variable
        step_reg = self._alloc_reg()
        self.instructions.append(Instruction(
            opcode=OpCode.MOV,
            dst=step_reg,
            src1=step_val,
        ))
        self.instructions.append(Instruction(
            opcode=OpCode.ADD,
            dst=loop_reg,
            src1=loop_reg,
            src2=step_reg,
        ))
        
        # Jump back to start
        self.instructions.append(Instruction(
            opcode=OpCode.JMP,
            dst=loop_start,
        ))
        
        # Loop end label
        self.instructions.append(Instruction(
            opcode=OpCode.NOP,
            label=loop_end,
        ))
    
    def _compile_while(self, stmt: ast.While):
        """Compile a while loop."""
        loop_start = self._new_label("while_start")
        loop_end = self._new_label("while_end")
        
        # Loop start label
        self.instructions.append(Instruction(
            opcode=OpCode.NOP,
            label=loop_start,
        ))
        
        # Compile condition
        cond_reg = self._compile_expr(stmt.test)
        
        # Branch to end if condition is false
        self.instructions.append(Instruction(
            opcode=OpCode.BRA_N,
            dst=loop_end,
            src1=cond_reg,
        ))
        
        # Compile loop body
        for s in stmt.body:
            self._compile_stmt(s)
        
        # Jump back to start
        self.instructions.append(Instruction(
            opcode=OpCode.JMP,
            dst=loop_start,
        ))
        
        # Loop end label
        self.instructions.append(Instruction(
            opcode=OpCode.NOP,
            label=loop_end,
        ))
