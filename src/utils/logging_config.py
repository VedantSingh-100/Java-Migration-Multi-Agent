import os
from datetime import datetime
from loguru import logger


def setup_migration_logging(repo_name: str):
    """
    Setup structured logging for migration process

    Creates three log files per repository:
    - llm_interactions.log: LLM prompts, responses, errors
    - multiagent_process.log: Agent coordination and workflow
    - summary.log: Commands, approaches, high-level decisions

    Args:
        repo_name: Repository name (e.g., "owner/repo")
    """
    # Remove default handler
    logger.remove()

    # Create safe repo name for filesystem
    repo_safe_name = repo_name.replace("/", "__").replace("\\", "__")

    # Create directory structure: logs/repo_name/
    repo_log_dir = os.path.join("logs", repo_safe_name)
    os.makedirs(repo_log_dir, exist_ok=True)

    # Generate timestamp for this migration session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console handler with minimal info
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n",
        colorize=True,
        filter=lambda record: record["extra"].get("log_type") in ["console", "summary"] or "log_type" not in record["extra"]
    )

    # 1. LLM Interactions Log
    llm_log_file = os.path.join(repo_log_dir, f"llm_interactions_{timestamp}.log")
    logger.add(
        sink=llm_log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        filter=lambda record: record["extra"].get("log_type") == "llm"
    )

    # 2. Multi-agent Process Log
    agent_log_file = os.path.join(repo_log_dir, f"multiagent_process_{timestamp}.log")
    logger.add(
        sink=agent_log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        filter=lambda record: record["extra"].get("log_type") == "agent"
    )

    # 3. Summary Log (commands, approaches, decisions)
    summary_log_file = os.path.join(repo_log_dir, f"summary_{timestamp}.log")
    logger.add(
        sink=summary_log_file,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        filter=lambda record: record["extra"].get("log_type") == "summary"
    )

    # Log the setup
    logger.bind(log_type="summary").info("=" * 80)
    logger.bind(log_type="summary").info(f"MIGRATION SESSION STARTED: {repo_name}")
    logger.bind(log_type="summary").info(f"Timestamp: {timestamp}")
    logger.bind(log_type="summary").info(f"Log directory: {repo_log_dir}")
    logger.bind(log_type="summary").info("=" * 80)

    return {
        "llm_log": llm_log_file,
        "agent_log": agent_log_file,
        "summary_log": summary_log_file,
        "log_dir": repo_log_dir
    }


# Convenience functions for different log types
def log_llm(message: str, level: str = "INFO"):
    """Log LLM-related messages"""
    getattr(logger.bind(log_type="llm"), level.lower())(message)


def log_agent(message: str, level: str = "INFO"):
    """Log multi-agent process messages"""
    getattr(logger.bind(log_type="agent"), level.lower())(message)


def log_summary(message: str, level: str = "INFO"):
    """Log summary messages (commands, approaches, decisions)"""
    getattr(logger.bind(log_type="summary"), level.lower())(message)


def log_console(message: str, level: str = "INFO"):
    """Log to console only"""
    getattr(logger.bind(log_type="console"), level.lower())(message)