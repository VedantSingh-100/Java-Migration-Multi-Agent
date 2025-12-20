"""
Prompt Loader Utility
Loads prompts from YAML files for easy modification
"""

from __future__ import annotations
import yaml
import os
from pathlib import Path

# Function to get target Java version dynamically (not cached at import time)
def get_target_java_version_from_env() -> str:
    """Get target Java version from environment variable dynamically"""
    return os.environ.get("TARGET_JAVA_VERSION", "21")


class PromptLoader:
    """Utility to load and format prompts from YAML files"""

    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            self.prompts_dir = Path(__file__).parent
        else:
            self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, filename: str, key: str = None) -> str:
        """Load a prompt from a YAML file"""
        filepath = self.prompts_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if key:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {filename}")
            return data[key]

        # If no key specified, return the first string value found
        for value in data.values():
            if isinstance(value, str):
                return value

        raise ValueError(f"No string prompt found in {filename}")

    def format_prompt(self, filename: str, key: str = None, **kwargs) -> str:
        """Load and format a prompt with variables"""
        prompt = self.load_prompt(filename, key)
        return prompt.format(**kwargs)

    def get_supervisor_prompt(self, target_java_version: str = None) -> str:
        """Get supervisor prompt"""
        version = target_java_version or get_target_java_version_from_env()
        prompt = self.load_prompt("supervisor.yaml", "supervisor_prompt")
        try:
            return prompt.format(target_java_version=version)
        except KeyError:
            return prompt

    def get_migration_request(self, project_path: str, target_java_version: str = None) -> str:
        """Get formatted migration request"""
        version = target_java_version or get_target_java_version_from_env()
        return self.format_prompt(
            "supervisor.yaml",
            "migration_request_template",
            project_path=project_path,
            target_java_version=version
        )

    def get_analysis_expert_prompt(self, target_java_version: str = None) -> str:
        """Get analysis expert prompt as string for create_react_agent"""
        version = target_java_version or get_target_java_version_from_env()
        prompt = self.load_prompt("analysis_expert.yaml", "analysis_expert_prompt")
        # Format with target_java_version if placeholder exists
        try:
            return prompt.format(target_java_version=version)
        except KeyError:
            return prompt

    def get_execution_expert_prompt(self, target_java_version: str = None) -> str:
        """Get execution expert prompt as string for create_react_agent"""
        version = target_java_version or get_target_java_version_from_env()
        prompt = self.load_prompt("execution_expert.yaml", "execution_expert_prompt")
        # Format with target_java_version if placeholder exists
        try:
            return prompt.format(target_java_version=version)
        except KeyError:
            return prompt

    def get_error_expert_prompt(self, target_java_version: str = None) -> str:
        """Get error expert prompt as string for create_react_agent"""
        version = target_java_version or get_target_java_version_from_env()
        prompt = self.load_prompt("error_expert.yaml", "error_expert_prompt")
        # Format with target_java_version if placeholder exists
        try:
            return prompt.format(target_java_version=version)
        except KeyError:
            return prompt


# Global prompt loader instance
prompt_loader = PromptLoader()


# Convenience functions
def get_supervisor_prompt() -> str:
    return prompt_loader.get_supervisor_prompt()


def get_migration_request(project_path: str) -> str:
    return prompt_loader.get_migration_request(project_path)


def get_analysis_expert_prompt(target_java_version: str = None) -> str:
    return prompt_loader.get_analysis_expert_prompt(target_java_version)


def get_execution_expert_prompt(target_java_version: str = None) -> str:
    return prompt_loader.get_execution_expert_prompt(target_java_version)


def get_error_expert_prompt(target_java_version: str = None) -> str:
    return prompt_loader.get_error_expert_prompt(target_java_version)


def get_target_java_version() -> str:
    """Get the configured target Java version"""
    return get_target_java_version_from_env()