from .random_policy import RandomPolicy
from .llm_policy import LLMPolicy

NAME_TO_POLICY = {
    "random": RandomPolicy,
    "llm": LLMPolicy
}