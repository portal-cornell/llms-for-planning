from .random_policy import RandomPolicy
from .llm_policy import LLMPolicy
from .ReAct_policy import ReActPolicy
from .backtrack_policy import BacktrackPolicy
from .lazy_policy import LazyPolicy
from .lazy_robotouille_policy import LazyRobotouillePolicy
from .tot_bfs_policy import ToTBFSPolicy
from .tot_dfs_policy import ToTDFSPolicy
from .tot_dfs_robotouille_policy import ToTDFSRobotouillePolicy

# Modify this dictionary to register a custom policy
NAME_TO_POLICY = {
    "random": RandomPolicy,
    "llm": LLMPolicy,
    "ReAct": ReActPolicy,
    "backtrack": BacktrackPolicy,
    "lazy": LazyPolicy,
    "lazy_robotouille": LazyRobotouillePolicy,
    "tot_bfs": ToTBFSPolicy,
    "tot_dfs": ToTDFSPolicy,
    "tot_dfs_robotouille": ToTDFSRobotouillePolicy
}