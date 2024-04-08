from .random_policy import RandomPolicy
from .llm_policy import LLMPolicy

# Modify this dictionary to register a custom policy
NAME_TO_POLICY = {
    "random": RandomPolicy,
    "llm": LLMPolicy
}