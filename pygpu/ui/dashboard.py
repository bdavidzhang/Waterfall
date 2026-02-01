"""
GPU Dashboard - A Textual-based terminal UI for visualizing GPU execution.

This provides a live visualization of:
- Thread states (active, waiting, masked, finished)
- Memory access patterns (heatmap)
- Execution progress
"""

from typing import Optional
import asyncio

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Label, ProgressBar
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.table import Table
from rich.panel import Panel

from pygpu.gpu import GPU


class ThreadGridWidget(Static):
    """Widget displaying a grid of threads with color-coded states."""
    
    THREAD_COLORS = {
        "ACTIVE": "green",
        "WAITING": "red",
        "MASKED": "dim white",
        "FINISHED": "blue",
    }
    
    THREAD_SYMBOLS = {
        "ACTIVE": "█",
        "WAITING": "▓",
        "MASKED": "░",
        "FINISHED": "▪",
    }
    
    def __init__(self, block_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.block_id = block_id
        self._grid: list[list[str]] = []
    
    def update_grid(self, grid: list[list[str]]):
        """Update the thread grid data."""
        self._grid = grid
        self.refresh()
    
    def render(self) -> Text:
        """Render the thread grid."""
        if not self._grid:
            return Text("No threads")
        
        text = Text()
        text.append(f"Block {self.block_id}\n", style="bold cyan")
        text.append("─" * (len(self._grid[0]) * 2 + 3) + "\n", style="dim")
        
        for warp_idx, row in enumerate(self._grid):
            text.append(f"W{warp_idx}│", style="dim")
            for state in row:
                color = self.THREAD_COLORS.get(state, "white")
                symbol = self.THREAD_SYMBOLS.get(state, "?")
                text.append(f"{symbol} ", style=color)
            text.append("\n")
        
        # Legend
        text.append("─" * (len(self._grid[0]) * 2 + 3) + "\n", style="dim")
        for state, color in self.THREAD_COLORS.items():
            symbol = self.THREAD_SYMBOLS[state]
            text.append(f" {symbol}", style=color)
            text.append(f"={state[:3]} ", style="dim")
        
        return text


class MemoryHeatmapWidget(Static):
    """Widget displaying memory access patterns as a heatmap."""
    
    HEAT_CHARS = " ░▒▓█"
    
    def __init__(self, width: int = 64, **kwargs):
        super().__init__(**kwargs)
        self._width = width
        self._reads: set[int] = set()
        self._writes: set[int] = set()
        self._range_start = 0
        self._range_end = 256
    
    def update_access(self, reads: list[int], writes: list[int]):
        """Update the memory access data."""
        self._reads = set(reads)
        self._writes = set(writes)
        self.refresh()
    
    def set_range(self, start: int, end: int):
        """Set the memory range to display."""
        self._range_start = start
        self._range_end = end
        self.refresh()
    
    def render(self) -> Text:
        """Render the memory heatmap."""
        text = Text()
        text.append("Memory Access\n", style="bold cyan")
        text.append(f"Range: [{self._range_start}:{self._range_end}]\n", style="dim")
        
        # Create a simplified view
        range_size = self._range_end - self._range_start
        cells_per_row = min(32, range_size)
        rows = (range_size + cells_per_row - 1) // cells_per_row
        
        for row in range(min(rows, 8)):  # Max 8 rows
            start = self._range_start + row * cells_per_row
            end = min(start + cells_per_row, self._range_end)
            
            text.append(f"{start:4d}│", style="dim")
            for addr in range(start, end):
                if addr in self._writes:
                    text.append("█", style="red")
                elif addr in self._reads:
                    text.append("▓", style="green")
                else:
                    text.append("░", style="dim")
            text.append("\n")
        
        # Legend
        text.append("\n", style="dim")
        text.append(" █", style="red")
        text.append("=Write ", style="dim")
        text.append("▓", style="green")
        text.append("=Read ", style="dim")
        text.append("░", style="dim")
        text.append("=Idle", style="dim")
        
        return text


class StatsWidget(Static):
    """Widget displaying GPU statistics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cycle = 0
        self._active_blocks = 0
        self._finished = False
        self._sm_stats: list[dict] = []
    
    def update_stats(self, state: dict):
        """Update the statistics from GPU state."""
        self._cycle = state.get("cycle", 0)
        self._finished = state.get("finished", False)
        self._sm_stats = state.get("sms", [])
        
        # Count active blocks
        self._active_blocks = sum(
            not b.get("finished", True)
            for b in state.get("blocks", [])
        )
        self.refresh()
    
    def render(self) -> Text:
        """Render the statistics."""
        text = Text()
        text.append("GPU Statistics\n", style="bold cyan")
        text.append("─" * 30 + "\n", style="dim")
        
        status = "FINISHED" if self._finished else "RUNNING"
        status_color = "green" if self._finished else "yellow"
        
        text.append(f"Status: ", style="dim")
        text.append(f"{status}\n", style=status_color)
        text.append(f"Cycle:  ", style="dim")
        text.append(f"{self._cycle}\n", style="white")
        text.append(f"Active Blocks: ", style="dim")
        text.append(f"{self._active_blocks}\n\n", style="white")
        
        # SM statistics
        text.append("Streaming Multiprocessors:\n", style="bold")
        for sm in self._sm_stats:
            sm_id = sm.get("sm_id", 0)
            util = sm.get("utilization", 0) * 100
            active = sm.get("active_blocks", 0)
            completed = sm.get("blocks_completed", 0)
            
            text.append(f"  SM{sm_id}: ", style="cyan")
            text.append(f"{util:5.1f}% ", style="yellow")
            text.append(f"[{active} active, {completed} done]\n", style="dim")
        
        return text


class GPUDashboard(App):
    """
    The main GPU visualization dashboard.
    
    Provides real-time visualization of GPU execution with:
    - Thread grid showing execution states
    - Memory access heatmap
    - Statistics and progress
    - Step-by-step or continuous execution
    """
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    
    #thread-container {
        column-span: 1;
        row-span: 2;
        border: solid green;
        padding: 1;
    }
    
    #memory-container {
        border: solid red;
        padding: 1;
    }
    
    #stats-container {
        border: solid blue;
        padding: 1;
    }
    
    #controls {
        dock: bottom;
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    
    Button {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        ("space", "step", "Step"),
        ("r", "run", "Run"),
        ("p", "pause", "Pause"),
        ("q", "quit", "Quit"),
    ]
    
    running = reactive(False)
    
    def __init__(self, gpu: GPU, **kwargs):
        super().__init__(**kwargs)
        self.gpu = gpu
        self._timer: Optional[Timer] = None
        self._speed = 0.1  # Seconds per cycle
    
    def compose(self) -> ComposeResult:
        """Create the dashboard layout."""
        yield Header()
        
        with Container(id="thread-container"):
            yield Label("Thread States", id="thread-label")
            yield ScrollableContainer(id="thread-grids")
        
        with Container(id="memory-container"):
            yield MemoryHeatmapWidget(id="memory-heatmap")
        
        with Container(id="stats-container"):
            yield StatsWidget(id="stats")
        
        with Horizontal(id="controls"):
            yield Button("Step [Space]", id="step-btn", variant="primary")
            yield Button("Run [R]", id="run-btn", variant="success")
            yield Button("Pause [P]", id="pause-btn", variant="warning")
            yield Button("Reset", id="reset-btn", variant="error")
        
        yield Footer()
    
    def on_mount(self):
        """Initialize the dashboard when mounted."""
        self._create_thread_widgets()
        self._update_display()
    
    def _create_thread_widgets(self):
        """Create thread grid widgets for each block."""
        container = self.query_one("#thread-grids", ScrollableContainer)
        
        for block in self.gpu.blocks:
            widget = ThreadGridWidget(
                block_id=block.block_id,
                id=f"block-{block.block_id}",
            )
            container.mount(widget)
    
    def _update_display(self):
        """Update all display widgets with current GPU state."""
        state = self.gpu.get_state()
        
        # Update thread grids
        for block_state in state.get("blocks", []):
            block_id = block_state["block_id"]
            try:
                widget = self.query_one(f"#block-{block_id}", ThreadGridWidget)
                widget.update_grid(block_state["thread_grid"])
            except Exception:
                pass
        
        # Update memory heatmap
        try:
            heatmap = self.query_one("#memory-heatmap", MemoryHeatmapWidget)
            heatmap.update_access(
                state.get("memory_reads", []),
                state.get("memory_writes", []),
            )
        except Exception:
            pass
        
        # Update stats
        try:
            stats = self.query_one("#stats", StatsWidget)
            stats.update_stats(state)
        except Exception:
            pass
    
    def action_step(self):
        """Execute a single step."""
        if not self.gpu.is_busy:
            return
        
        self.gpu.tick()
        self._update_display()
    
    def action_run(self):
        """Start continuous execution."""
        if self.running:
            return
        
        self.running = True
        self._timer = self.set_interval(self._speed, self._auto_step)
    
    def action_pause(self):
        """Pause execution."""
        self.running = False
        if self._timer:
            self._timer.stop()
            self._timer = None
    
    def _auto_step(self):
        """Auto-step callback for continuous execution."""
        if not self.gpu.is_busy:
            self.action_pause()
            return
        
        self.gpu.tick()
        self._update_display()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "step-btn":
            self.action_step()
        elif button_id == "run-btn":
            self.action_run()
        elif button_id == "pause-btn":
            self.action_pause()
        elif button_id == "reset-btn":
            self._reset_simulation()
    
    def _reset_simulation(self):
        """Reset the simulation to the beginning."""
        self.action_pause()
        # Re-launch with same configuration
        if self.gpu.program and self.gpu.blocks:
            num_blocks = len(self.gpu.blocks)
            threads = self.gpu.blocks[0].num_threads if self.gpu.blocks else 8
            self.gpu.launch(blocks=num_blocks, threads_per_block=threads)
            self._update_display()


def run_dashboard(gpu: GPU):
    """
    Run the GPU dashboard for a configured GPU.
    
    Usage:
        gpu = GPU()
        gpu.load_program(my_kernel)
        gpu.launch(blocks=2, threads_per_block=8)
        run_dashboard(gpu)
    """
    app = GPUDashboard(gpu)
    app.run()
