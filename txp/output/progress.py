"""Rich-based timeline progress display for agent execution."""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    
    @property
    def is_done(self) -> bool:
        return self in (AgentStatus.SUCCESS, AgentStatus.FAILED)


# Status display config: (icon, style)
STATUS_DISPLAY = {
    AgentStatus.PENDING: ("○", "dim"),
    AgentStatus.RUNNING: ("⋯", "cyan bold"),
    AgentStatus.SUCCESS: ("✓", "green"),
    AgentStatus.FAILED: ("✗", "red"),
}


@dataclass
class AgentState:
    """Tracks state of a single agent."""
    agent_id: int
    status: AgentStatus = AgentStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        if not self.start_time:
            return 0.0
        return (self.end_time or time.time()) - self.start_time


@dataclass 
class ProgressState:
    """Tracks overall progress state."""
    total: int
    agents: Dict[int, AgentState] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.agents = {i: AgentState(agent_id=i) for i in range(self.total)}
    
    def count_by_status(self, *statuses: AgentStatus) -> int:
        return sum(1 for a in self.agents.values() if a.status in statuses)
    
    @property
    def completed(self) -> int:
        return self.count_by_status(AgentStatus.SUCCESS, AgentStatus.FAILED)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_duration(self) -> float:
        done = [a for a in self.agents.values() if a.status.is_done]
        return sum(a.duration for a in done) / len(done) if done else 0.0


class TimelineProgress:
    """Rich-based timeline progress display with auto-refresh."""
    
    def __init__(self, total_agents: int, max_concurrency: Optional[int] = None):
        self.state = ProgressState(total=total_agents)
        self.max_concurrency = max_concurrency
        self.console = Console()
        self._live: Optional[Live] = None
        self._stopped = False
    
    def __rich__(self) -> Panel:
        """Rich protocol for auto-refresh."""
        return self._build_panel()
    
    def _build_agent_line(self, agent: AgentState) -> Text:
        icon, style = STATUS_DISPLAY[agent.status]
        line = Text()
        line.append(f"  {icon} Agent {agent.agent_id:2d} ", style=style if agent.status != AgentStatus.RUNNING else "bold")
        
        bar_width = 30
        if agent.status == AgentStatus.RUNNING:
            # Animated pulse effect
            pulse = int((agent.duration * 3) % (bar_width * 2))
            pulse = bar_width * 2 - pulse if pulse > bar_width else pulse
            line.append("━" * pulse + "░" * (bar_width - pulse), style="cyan")
            line.append(f" {agent.duration:.1f}s", style="cyan bold")
        elif agent.status.is_done:
            color = "green" if agent.status == AgentStatus.SUCCESS else "red"
            line.append("━" * bar_width, style=f"{color} dim")
            line.append(f" {agent.duration:.1f}s", style=color)
            if agent.error:
                err = agent.error[:40] + "..." if len(agent.error) > 40 else agent.error
                line.append(f"\n      └─ {err}", style="red dim")
        else:  # PENDING
            line.append("░" * bar_width + " waiting", style="dim")
        
        return line
    
    def _build_panel(self) -> Panel:
        # Sort: running → completed (by end time) → pending
        def sort_key(a: AgentState) -> tuple:
            if a.status == AgentStatus.RUNNING:
                return (0, a.start_time or 0)
            if a.status.is_done:
                return (1, a.end_time or 0)
            return (2, a.agent_id)
        
        lines = [self._build_agent_line(a) for a in sorted(self.state.agents.values(), key=sort_key)]
        
        # Summary line
        s = self.state
        pct = s.completed / s.total if s.total else 0
        bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
        
        summary = Text(f"\n  {bar} {s.completed}/{s.total}  │  ", style="bold")
        for status, label in [(AgentStatus.SUCCESS, "✓"), (AgentStatus.FAILED, "✗"), (AgentStatus.RUNNING, "⋯")]:
            count = s.count_by_status(status)
            if count:
                summary.append(f"{label}{count} ", style=STATUS_DISPLAY[status][1].split()[0])
        summary.append(f" │  {s.elapsed:.1f}s", style="bold")
        if s.avg_duration:
            summary.append(f"  │  avg: {s.avg_duration:.1f}s", style="dim")
        
        lines.append(summary)
        
        concurrency = f" ({self.max_concurrency} concurrent)" if self.max_concurrency else ""
        return Panel(Group(*lines), title=f"[bold]Spawning {s.total} agents{concurrency}[/bold]", border_style="blue")
    
    def start(self) -> None:
        self._live = Live(self, console=self.console, refresh_per_second=4, transient=False)
        self._live.start()
    
    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._live:
            self._live.stop()
            self._live = None
            self.console.print()
    
    def agent_started(self, agent_id: int) -> None:
        if agent_id in self.state.agents:
            self.state.agents[agent_id].status = AgentStatus.RUNNING
            self.state.agents[agent_id].start_time = time.time()
    
    def agent_completed(self, agent_id: int, success: bool, error: Optional[str] = None) -> None:
        if agent_id in self.state.agents:
            a = self.state.agents[agent_id]
            a.status = AgentStatus.SUCCESS if success else AgentStatus.FAILED
            a.end_time = time.time()
            a.error = error
    
    def get_callbacks(self) -> tuple[Callable[[int], None], Callable[[int, bool, Optional[str]], None]]:
        return self.agent_started, self.agent_completed


def create_progress_display(
    total_agents: int,
    quiet: bool = False,
    max_concurrency: Optional[int] = None,
) -> Optional[TimelineProgress]:
    """Create progress display. Returns None if quiet or non-TTY."""
    if quiet or not sys.stdout.isatty():
        return None
    return TimelineProgress(total_agents, max_concurrency=max_concurrency)
