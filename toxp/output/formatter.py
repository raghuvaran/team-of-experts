"""Output formatting with colors, progress indicators, and streaming support."""

import sys
from typing import Optional, TextIO


# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


class OutputFormatter:
    """Handles CLI output with colors, progress indicators, and streaming.
    
    Supports quiet mode (only final answer), verbose mode (detailed debug info),
    and colored output for distinguishing errors, warnings, and success messages.
    """
    
    def __init__(
        self,
        quiet: bool = False,
        verbose: bool = False,
        use_color: bool = True,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ):
        """Initialize the output formatter.
        
        Args:
            quiet: If True, only display the final answer (suppress info messages).
            verbose: If True, display detailed debug information.
            use_color: If True and output is a TTY, use colored output.
            stdout: Output stream for normal messages (defaults to sys.stdout).
            stderr: Output stream for error messages (defaults to sys.stderr).
        """
        self.quiet = quiet
        self.verbose = verbose
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self.use_color = use_color and self._stdout.isatty()
        self._progress_shown = False
    
    def info(self, message: str) -> None:
        """Print info message (skipped in quiet mode).
        
        Args:
            message: The message to display.
        """
        if not self.quiet:
            self._clear_progress_line()
            print(message, file=self._stdout)
    
    def debug(self, message: str) -> None:
        """Print debug message (only shown in verbose mode).
        
        Args:
            message: The debug message to display.
        """
        if self.verbose:
            self._clear_progress_line()
            self._print_colored(f"[DEBUG] {message}", CYAN)
    
    def success(self, message: str) -> None:
        """Print success message in green.
        
        Args:
            message: The success message to display.
        """
        if not self.quiet:
            self._clear_progress_line()
            self._print_colored(message, GREEN)
    
    def warning(self, message: str) -> None:
        """Print warning message in yellow.
        
        Args:
            message: The warning message to display.
        """
        self._clear_progress_line()
        self._print_colored(message, YELLOW)
    
    def error(self, message: str) -> None:
        """Print error message in red to stderr.
        
        Args:
            message: The error message to display.
        """
        self._clear_progress_line()
        self._print_colored(message, RED, file=self._stderr)
    
    def stream_token(self, token: str) -> None:
        """Stream a single token to stdout without newline.
        
        Used for real-time streaming of coordinator synthesis output.
        
        Args:
            token: The token to output.
        """
        print(token, end="", flush=True, file=self._stdout)
    
    def stream_end(self) -> None:
        """End streaming output with a newline."""
        print(file=self._stdout)
    
    def progress(self, current: int, total: int, message: str = "") -> None:
        """Display progress indicator showing agent execution status.
        
        Shows a progress bar with current/total count and optional message.
        Skipped in quiet mode.
        
        Args:
            current: Current progress count.
            total: Total count for completion.
            message: Optional status message to display.
        """
        if self.quiet:
            return
        
        # Calculate progress bar
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "=" * filled + " " * (bar_width - filled)
        
        # Build progress line
        progress_line = f"\r[{bar}] {current}/{total}"
        if message:
            progress_line += f" {message}"
        
        # Clear to end of line and print
        print(f"{progress_line}\033[K", end="", flush=True, file=self._stdout)
        self._progress_shown = True
    
    def progress_complete(self, message: str = "") -> None:
        """Complete progress indicator and move to new line.
        
        Args:
            message: Optional completion message to display.
        """
        if self.quiet:
            return
        
        if self._progress_shown:
            print(file=self._stdout)  # Move to new line
            self._progress_shown = False
        
        if message:
            self.success(message)
    
    def confidence(self, level: str) -> None:
        """Display confidence level with appropriate color.
        
        Args:
            level: Confidence level ("Low", "Medium", or "High").
        """
        color = {
            "Low": RED,
            "Medium": YELLOW,
            "High": GREEN,
        }.get(level, RESET)
        
        self._clear_progress_line()
        if self.use_color:
            print(f"\n{BOLD}Confidence:{RESET} {color}{level}{RESET}", file=self._stdout)
        else:
            print(f"\nConfidence: {level}", file=self._stdout)
    
    def final_answer(self, answer: str, confidence_level: Optional[str] = None) -> None:
        """Display the final synthesized answer prominently.
        
        This is always shown, even in quiet mode.
        
        Args:
            answer: The final answer text.
            confidence_level: Optional confidence level to display.
        """
        self._clear_progress_line()
        
        if confidence_level:
            self.confidence(confidence_level)
        
        if not self.quiet:
            print(file=self._stdout)  # Blank line before answer
        
        print(answer, file=self._stdout)
    
    def agent_summary(self, successful: int, total: int) -> None:
        """Display agent success/failure summary.
        
        Args:
            successful: Number of successful agents.
            total: Total number of agents.
        """
        if self.quiet:
            return
        
        self._clear_progress_line()
        
        failed = total - successful
        if failed == 0:
            self.success(f"✓ All {total} agents completed successfully")
        elif successful >= total // 2:
            self.warning(f"⚠ {successful}/{total} agents succeeded ({failed} failed)")
        else:
            self.error(f"✗ Only {successful}/{total} agents succeeded")
    
    def _print_colored(
        self,
        message: str,
        color: str,
        file: Optional[TextIO] = None,
    ) -> None:
        """Print message with optional color.
        
        Args:
            message: The message to print.
            color: ANSI color code to use.
            file: Output file (defaults to stdout).
        """
        output_file = file or self._stdout
        if self.use_color:
            print(f"{color}{message}{RESET}", file=output_file)
        else:
            print(message, file=output_file)
    
    def _clear_progress_line(self) -> None:
        """Clear the progress line if one is shown."""
        if self._progress_shown:
            print("\r\033[K", end="", file=self._stdout)
            self._progress_shown = False
