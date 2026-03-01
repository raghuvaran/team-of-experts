"""TOXP CLI - Command-line interface for the Team of eXPerts system.

Feature: toxp-cli
Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 9.3, 9.8, 9.9
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from toxp import __version__
from toxp.config import ConfigManager, _denormalize_key
from toxp.exceptions import (
    ToxpError,
    ConfigurationError,
    CredentialsError,
    ProviderError,
    ModelNotFoundError,
    ThrottlingError,
    InsufficientAgentsError,
    NetworkError,
    TimeoutError as ToxpTimeoutError,
)
from toxp.logging.session_logger import SessionLogger
from toxp.output.formatter import OutputFormatter
from toxp.output.progress import create_progress_display


class CLICallbacks:
    """QueryCallbacks implementation that drives Rich progress + streaming."""

    def __init__(self, formatter: OutputFormatter, progress):
        self._formatter = formatter
        self._progress = progress
        self._agent_start_cb = None
        self._agent_complete_cb = None
        self._agent_token_cb = None
        if progress:
            (
                self._agent_start_cb,
                self._agent_complete_cb,
                self._agent_token_cb,
            ) = progress.get_callbacks()

    def on_agent_start(self, agent_id: int) -> None:
        if self._agent_start_cb:
            self._agent_start_cb(agent_id)

    def on_agent_complete(
        self, agent_id: int, success: bool, error: Optional[str] = None
    ) -> None:
        if self._agent_complete_cb:
            self._agent_complete_cb(agent_id, success, error)

    def on_agent_token(self, agent_id: int, token: str) -> None:
        if self._agent_token_cb:
            self._agent_token_cb(agent_id, token)

    def on_agents_done(self) -> None:
        if self._progress:
            self._progress.stop()

    def on_coordinator_token(self, token: str) -> None:
        self._formatter.stream_token(token)


def _add_query_flags(parser: argparse.ArgumentParser) -> None:
    """Add query-mode flags to a parser. Single source of truth for all query flags."""
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-q", "--query", dest="query_flag", metavar="TEXT",
                        help="Query text to process")
    parser.add_argument("-o", "--output", dest="output_file", metavar="FILE",
                        help="Write final answer to FILE")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output including debug info and stack traces on errors")
    parser.add_argument("-Q", "--quiet", action="store_true",
                        help="Only show the final answer (no progress or status)")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip session logging for this execution")
    parser.add_argument("-n", "--num-agents", type=int, metavar="N",
                        help="Number of reasoning agents to spawn (2-32, default: 15)")
    parser.add_argument("-m", "--model", metavar="ID",
                        help="Model ID to use (e.g., "
                        "us.anthropic.claude-sonnet-4-5-20250929-v1:0)")
    parser.add_argument("--temperature", type=float, metavar="T",
                        help="Sampling temperature for agents (0.0-1.0, default: 0.9)")
    parser.add_argument("--aws-profile", metavar="PROFILE",
                        help="AWS profile for credentials (default: default)")
    parser.add_argument("--region", metavar="REGION",
                        help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("-c", "--max-concurrency", type=int, metavar="N",
                        help="Maximum concurrent API requests "
                        "(default: auto-calculated based on model quotas)")
    parser.add_argument("--context-1m", action="store_true", default=None,
                        help="Enable 1M token context window beta")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all commands and options.

    Requirements: 10.5 - Provides --help documentation for all commands and options.
    """
    parser = argparse.ArgumentParser(
        prog="toxp",
        description="""
TOXP - Team Of eXPerts parallel reasoning system

TOXP spawns multiple independent reasoning agents to tackle complex queries,
then synthesizes their outputs through a coordinator agent into a coherent,
high-confidence answer.
        """.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  toxp "What is the capital of France?"     Query with positional argument
  toxp -q "Explain quantum computing"       Query with --query flag
  echo "Hello" | toxp                        Query from stdin
  toxp config show                           Show all configuration
  toxp config set num-agents 8               Set configuration value
  toxp config get model                      Get a specific config value
  toxp config reset                          Reset to default configuration

Configuration:
  Configuration is stored at ~/.toxp/config.json
  Environment variable TOXP_AWS_PROFILE overrides aws-profile config
  CLI arguments override environment variables and config file

For more information, visit: https://github.com/your-repo/team-of-experts
        """,
    )

    _add_query_flags(parser)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config subcommand
    config_parser = subparsers.add_parser("config",
                                          help="Manage TOXP configuration",
                                          description="Manage TOXP configuration stored at "
                                          "~/.toxp/config.json")
    config_sub = config_parser.add_subparsers(dest="config_action", help="Config actions")

    set_p = config_sub.add_parser("set", help="Set a configuration value",
                                  description="Set a configuration value in "
                                  "~/.toxp/config.json")
    set_p.add_argument("key", help="Configuration key (e.g., num-agents, model, aws-profile)")
    set_p.add_argument("value", help="Value to set")

    get_p = config_sub.add_parser("get", help="Get a configuration value",
                                  description="Get the current value of a configuration key")
    get_p.add_argument("key", help="Configuration key to retrieve")

    config_sub.add_parser("show", help="Show all configuration values",
                          description="Display all current configuration values")
    config_sub.add_parser("path", help="Show config file path",
                          description="Display the path to the configuration file")
    config_sub.add_parser("reset", help="Reset to default configuration",
                          description="Reset all configuration values to their defaults")

    return parser



def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse arguments, returning (args, remaining positional args)."""
    if argv is None:
        argv = sys.argv[1:]

    # Show full help with subcommands when --help or no args
    if not argv or argv[0] in ["-h", "--help"]:
        parser = create_parser()
        return parser.parse_known_args(argv)

    # Check if first arg is a subcommand
    if argv[0] == "config":
        parser = create_parser()
        return parser.parse_known_args(argv)

    # For query mode, create a parser without subcommands to avoid conflicts
    # Uses _add_query_flags to stay in sync with create_parser
    parser = argparse.ArgumentParser(
        prog="toxp",
        description="TOXP - Team Of eXPerts parallel reasoning system",
    )
    _add_query_flags(parser)

    args, remaining = parser.parse_known_args(argv)
    args.command = None  # Not a subcommand
    return args, remaining


def read_stdin_query() -> Optional[str]:
    """Read query from stdin if piped."""
    if sys.stdin.isatty():
        return None
    try:
        content = sys.stdin.read().strip()
        return content if content else None
    except Exception:
        return None


def get_query_text(args: argparse.Namespace, remaining: List[str]) -> Optional[str]:
    """Get query with precedence: --query > positional > stdin."""
    if args.query_flag:
        return args.query_flag
    if remaining:
        return " ".join(remaining)
    return read_stdin_query()


def handle_config_command(args: argparse.Namespace, formatter: OutputFormatter) -> int:
    """Handle config subcommand."""
    config_manager = ConfigManager()
    
    if args.config_action == "set":
        return handle_config_set(args, config_manager, formatter)
    elif args.config_action == "get":
        return handle_config_get(args, config_manager, formatter)
    elif args.config_action == "show":
        return handle_config_show(config_manager, formatter)
    elif args.config_action == "path":
        return handle_config_path(config_manager, formatter)
    elif args.config_action == "reset":
        return handle_config_reset(config_manager, formatter)
    else:
        formatter.error("No config action. Use: set, get, show, path, or reset")
        return 1


def handle_config_set(args: argparse.Namespace, config_manager: ConfigManager,
                      formatter: OutputFormatter) -> int:
    """Handle 'config set' command. Requirements: 2.2"""
    try:
        config_manager.set(args.key, args.value)
        formatter.success(f"Set {args.key} = {args.value}")
        return 0
    except KeyError as e:
        formatter.error(str(e))
        formatter.info(f"Valid keys: {', '.join(config_manager.get_valid_keys())}")
        return 1
    except ValueError as e:
        formatter.error(f"Invalid value: {e}")
        return 1


def handle_config_get(args: argparse.Namespace, config_manager: ConfigManager,
                      formatter: OutputFormatter) -> int:
    """Handle 'config get' command. Requirements: 2.3"""
    try:
        value = config_manager.get(args.key)
        print(value)
        return 0
    except KeyError as e:
        formatter.error(str(e))
        formatter.info(f"Valid keys: {', '.join(config_manager.get_valid_keys())}")
        return 1


def handle_config_show(config_manager: ConfigManager, formatter: OutputFormatter) -> int:
    """Handle 'config show' command. Requirements: 2.4"""
    config = config_manager.show()
    print("TOXP Configuration:")
    print("-" * 40)
    for key, value in sorted(config.items()):
        display_key = _denormalize_key(key)
        print(f"  {display_key}: {value}")
    return 0


def handle_config_path(config_manager: ConfigManager, formatter: OutputFormatter) -> int:
    """Handle 'config path' command. Requirements: 2.5"""
    print(config_manager.config_path)
    return 0


def handle_config_reset(config_manager: ConfigManager, formatter: OutputFormatter) -> int:
    """Handle 'config reset' command. Requirements: 2.6"""
    config_manager.reset()
    formatter.success("Configuration reset to defaults")
    return 0



async def handle_query_command(args: argparse.Namespace, remaining: List[str],
                               formatter: OutputFormatter) -> int:
    """Handle query execution. Requirements: 3.1-3.5, 9.3, 9.8, 9.9, 10.1-10.8"""
    query_text = get_query_text(args, remaining)

    if not query_text:
        formatter.error("No query provided")
        formatter.info("Usage: toxp \"your question\" or echo \"question\" | toxp")
        formatter.info("Run 'toxp --help' for more information")
        return 1

    verbose = getattr(args, "verbose", False)
    log_enabled = not getattr(args, "no_log", False)

    # Build config overrides from CLI args
    cli_overrides = {
        k: v for k, v in {
            "num_agents": getattr(args, "num_agents", None),
            "model": getattr(args, "model", None),
            "temperature": getattr(args, "temperature", None),
            "aws_profile": getattr(args, "aws_profile", None),
            "region": getattr(args, "region", None),
            "max_concurrency": getattr(args, "max_concurrency", None),
            "context_1m": getattr(args, "context_1m", None),
        }.items() if v is not None
    }

    formatter.debug(f"Query: {query_text[:100]}...")

    try:
        from toxp.api import run_query, NoOpCallbacks
        from toxp.config import ConfigManager

        # Load config to read log settings and create progress display
        config_manager = ConfigManager()
        config = config_manager.load_with_overrides(cli_overrides or None)
        log_enabled = config.log_enabled and log_enabled

        formatter.debug(
            f"Config: num_agents={config.num_agents}, model={config.model}"
        )

        # Create progress display (returns None if quiet or non-TTY)
        progress = create_progress_display(
            total_agents=config.num_agents,
            quiet=args.quiet,
            max_concurrency=config.max_concurrency or config.num_agents,
        )

        # Wire CLI callbacks
        callbacks = CLICallbacks(formatter, progress)

        if progress:
            progress.start()

        try:
            result = await run_query(
                query_text,
                config_overrides=cli_overrides or None,
                callbacks=callbacks if not args.quiet else None,
            )
        finally:
            if progress:
                progress.stop()

        if not args.quiet:
            formatter.stream_end()

        formatter.agent_summary(result.successful_agents, result.total_agents)

        formatter.final_answer(
            result.final_answer,
            confidence_level=result.confidence,
        )

        if args.output_file:
            output_path = Path(args.output_file)
            output_path.write_text(result.final_answer)
            formatter.success(f"Answer written to {output_path}")

        if log_enabled and result.raw_result:
            session_logger = SessionLogger(
                enabled=True, retention_days=config.log_retention_days
            )
            log_path = session_logger.log_session(result.raw_result)
            if log_path:
                formatter.debug(f"Session logged to {log_path}")

        return 0

    except CredentialsError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ModelNotFoundError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ThrottlingError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except InsufficientAgentsError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except NetworkError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ToxpTimeoutError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ConfigurationError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ProviderError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        return 1

    except ToxpError as e:
        formatter.error(e.format_for_user(verbose=verbose))
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    except ValueError as e:
        formatter.error(f"Configuration error: {e}")
        return 1

    except KeyboardInterrupt:
        formatter.info("\nOperation cancelled by user")
        return 130

    except Exception as e:
        formatter.error(f"Unexpected error: {e}")
        if verbose:
            import traceback
            formatter.info("\nStack trace:")
            traceback.print_exc()
        else:
            formatter.info("Run with --verbose for more details")
        return 1


def main() -> int:
    """Main entry point for the TOXP CLI.
    
    Requirements: 10.4, 10.5, 10.8 - Error handling and help documentation.
    """
    try:
        args, remaining = parse_args()
        
        formatter = OutputFormatter(
            quiet=getattr(args, "quiet", False),
            verbose=getattr(args, "verbose", False),
        )
        
        if args.command == "config":
            return handle_config_command(args, formatter)
        
        return asyncio.run(handle_query_command(args, remaining, formatter))
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        # Last resort error handling
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
