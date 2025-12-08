import os
import pathlib

# Set tiktoken cache directory (will be created automatically if needed)
tiktoken_cache_dir = str(pathlib.Path(__file__).parent.parent.parent / "tiktoken_cache")
os.makedirs(tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    """Count prompt & response tokens using cl100k_base (Sonnet 3.7)."""

    # Pricing per 1M tokens for Claude 3.7 Sonnet (update as needed)
    COST_PER_MILLION_PROMPT = 3.00  # $3 per 1M input tokens
    COST_PER_MILLION_RESPONSE = 15.00  # $15 per 1M output tokens

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.llm_calls = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        for p in prompts:
            self.prompt_tokens += len(self.enc.encode(p))

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                # 🔹 prefer the explicit attribute for chat models
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    content = gen.message.content
                    # Handle content blocks (list) from newer Claude models
                    if isinstance(content, list):
                        text = " ".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in content
                        )
                    else:
                        text = content or ""
                # 🔹 fall-back for completion-style
                elif hasattr(gen, "text"):
                    text = gen.text or ""
                else:
                    text = ""
                # Ensure text is a string before encoding
                if not isinstance(text, str):
                    text = str(text) if text else ""
                self.response_tokens += len(self.enc.encode(text))

    def calculate_cost(self):
        """Calculate the cost based on token usage"""
        prompt_cost = (self.prompt_tokens / 1_000_000) * self.COST_PER_MILLION_PROMPT
        response_cost = (self.response_tokens / 1_000_000) * self.COST_PER_MILLION_RESPONSE
        total_cost = prompt_cost + response_cost
        return prompt_cost, response_cost, total_cost

    def get_stats(self):
        """Return token statistics as a dictionary"""
        prompt_cost, response_cost, total_cost = self.calculate_cost()
        total_tokens = self.prompt_tokens + self.response_tokens

        return {
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": total_tokens,
            "llm_calls": self.llm_calls,
            "prompt_cost_usd": prompt_cost,
            "response_cost_usd": response_cost,
            "total_cost_usd": total_cost
        }

    def report(self, log_func=None):
        """Print or log token usage report

        Args:
            log_func: Optional logging function (e.g., log_summary). If None, uses print.
        """
        stats = self.get_stats()
        prompt_cost, response_cost, total_cost = self.calculate_cost()

        output_func = log_func if log_func else print

        output_func("\n" + "=" * 60)
        output_func("TOKEN USAGE & COST REPORT")
        output_func("=" * 60)
        output_func(f"LLM Calls:        {stats['llm_calls']:,}")
        output_func(f"Prompt tokens:    {stats['prompt_tokens']:,}")
        output_func(f"Response tokens:  {stats['response_tokens']:,}")
        output_func(f"Total tokens:     {stats['total_tokens']:,}")
        output_func("-" * 60)
        output_func(f"Prompt cost:      ${prompt_cost:.4f}")
        output_func(f"Response cost:    ${response_cost:.4f}")
        output_func(f"Total cost:       ${total_cost:.4f}")
        output_func("=" * 60)

        return stats

    def reset(self):
        """Reset all counters"""
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.llm_calls = 0


# tc = TokenCounter()