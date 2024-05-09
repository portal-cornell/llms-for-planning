from .random_policy import RandomPolicy
from .llm_policy import LLMPolicy
from .ReAct_policy import ReActPolicy
from .backtrack_policy import BacktrackPolicy
from .lazy_policy import LazyPolicy
from .tot_bfs_policy import ToTBFSPolicy

# Modify this dictionary to register a custom policy
NAME_TO_POLICY = {
    "random": RandomPolicy,
    "llm": LLMPolicy,
    "ReAct": ReActPolicy,
    "backtrack": BacktrackPolicy,
    "lazy": LazyPolicy,
    "tot_bfs": ToTBFSPolicy
}