"""This is a command line interface for the news briefing service."""
import argparse
import asyncio
import logging
import sys
from src.config import settings
logger = logging.getLogger(__name__)

def configure_logging() -> None:
    """It configures the logging settings based on the log level specified in the settings."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def create_parser() -> argparse.ArgumentParser:
    """It creates a command line argument parser."""
    parser = argparse.ArgumentParser(prog="newsbrief", description="Generate personalized AI-powered daily news briefings.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_daily_parser = subparsers.add_parser("run-daily", help="Generate a daily news briefing for a user.")
    run_daily_parser.add_argument("--user", required=True, help="User ID for whom the daily briefing will be generated.")
    return parser

def parse_arguments() -> argparse.Namespace:
    """It parses the command line arguments."""
    parser = create_parser()
    return parser.parse_args()

async def run_daily(user_id: str) -> None:
    """It generates a daily news briefing for the specified user."""
    from src.concurrency.pipeline import run_daily_pipeline
    logger.info("Starting daily news briefing generation for user: %s", user_id)
    await run_daily_pipeline(user_id)
    logger.info("Completed daily news briefing generation for user: %s", user_id)

def main() -> int:
    """It runs the command-line interface and returns an exit status."""
    configure_logging()
    try:
        args = parse_arguments()
        if args.command == "run-daily":
            asyncio.run(run_daily(args.user))
            return 0
        logger.error("Unknown command: %s", args.command)
        return 2
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        return 130
    except Exception:
        logger.exception("Daily briefing failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


