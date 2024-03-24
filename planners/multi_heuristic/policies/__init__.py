from policies.random_policy import RandomPolicy
from policies.bfs_policy import BFSPolicy

NAME_TO_POLICY = {
    "random": RandomPolicy,
    "bfs": BFSPolicy
}