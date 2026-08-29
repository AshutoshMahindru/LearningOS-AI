# WP-139: Tutor and Model Provider Architecture

## 1. Provider-Agnostic Model Interface
LearningOS V3 interacts with language models exclusively through an abstract adapter interface:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, messages: list[dict], system_prompt: str, temperature: float = 0.2) -> str:
        pass
```

Supported Providers:
- **Local Provider**: Ollama / llama.cpp / vLLM (OpenAI-compatible `/v1/chat/completions`)
- **Remote Providers**: OpenAI, Anthropic, Google Gemini (via official SDKs or OpenAI-compatible format)

## 2. Tutor Roles & Dynamic System Prompts
The tutor dynamically switches roles based on the learner's context:
- **Navigator**: Clarifies mission goals, invariants, and sequence.
- **Socratic Tutor**: Asks targeted probing questions; never generates direct solutions.
- **Debugger**: Helps isolate error messages and stack traces.
- **Feynman Reviewer**: Tests whether learner explanations are free of jargon and logically sound.
