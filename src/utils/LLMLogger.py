from langchain_core.callbacks import BaseCallbackHandler
from src.utils.logging_config import log_llm, log_summary, log_console


class LLMLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        log_llm("LLM PROMPT START")
        log_llm(f"Model: {serialized.get('id', ['unknown'])}", "DEBUG")
        for i, prompt in enumerate(prompts):
            log_llm(f"--- PROMPT {i+1} ---", "DEBUG")
            log_llm(prompt, "DEBUG")
            log_llm("--- END PROMPT ---", "DEBUG")

    def on_llm_end(self, response, **kwargs):
        log_llm("LLM RESPONSE RECEIVED")

        if hasattr(response, 'generations') and response.generations:
            for i, generation in enumerate(response.generations):
                log_llm(f"--- RESPONSE {i+1} ---", "DEBUG")
                for j, choice in enumerate(generation):
                    log_llm(f"Choice {j+1}: {choice.text}", "DEBUG")
                log_llm("--- END RESPONSE ---", "DEBUG")

    def on_llm_error(self, error, **kwargs):
        # Log error to both LLM log and summary
        log_llm(f"LLM ERROR: {error}", "ERROR")
        log_summary(f"LLM Error: {error}", "ERROR")
        log_console(f"LLM Error: {error}", "ERROR")