# Java-Migration-Multi-Agent

Automated Java ecosystem migration framework using multi-agent LLM orchestration.

## Target Migrations

- Java 8+ → Java 21
- Spring Boot 2.x → 3.x
- Spring Framework 5.x → 6.x
- javax.* → jakarta.* namespace
- JUnit 4 → JUnit 5

## Technology Stack

- Python
- LangGraph / LangChain
- OpenAI API (Claude via Vertex AI)
- OpenRewrite

## Architecture

Multi-agent orchestration with specialized agents:

- **Supervisor**: Routes between agents, validates progress
- **Analysis Expert**: POM analysis, dependency mapping, creates migration plan
- **Execution Expert**: Executes migration tasks, runs OpenRewrite recipes
- **Error Expert**: Diagnoses compilation/test failures, applies fixes

## Usage

```bash
python migrate_single_repo.py <repo_name> <base_commit> [--csv <path>]
```

## Project Structure

```
├── migrate_single_repo.py          # CLI entry point
├── supervisor_orchestrator_refactored.py  # Main orchestration engine
├── prompts/                        # Agent prompt YAML files
├── src/
│   ├── tools/                      # Migration tools (52+ tools)
│   ├── orchestrator/               # Modular orchestration components
│   └── utils/                      # Utility modules
└── logs/                           # Migration logs
```

## Key Features

- Circuit breaker for LLM cost control
- Token usage tracking
- Context compression and management
- Loop/stuck detection
- State file protection
- Self-healing mechanisms

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive agent documentation.
