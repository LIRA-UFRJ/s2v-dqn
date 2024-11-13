import sys
sys.path.append("../")  # go to parent dir

from s2v_dqn.graph_type import GraphType
from s2v_dqn.interaction import train


PROBLEM = 'MVC'

mvc_base_params = {
    'embedding_layers': 5,
    'nstep': 2,
    'lr_config': [[0, 1e-3], [0.5, 1e-4], [0.75, 1e-5]],
    'eps_start': 1.00,
    'eps_end': 0.00,
    'eps_end_at_episode': 1.0,
    'n_episodes': 5000,
    'validate_each': 100,
    'print_train_metrics_each': 10,
    'batch_size': 64,
    'exact_solution_max_size': 20,
}
params = {
    **mvc_base_params,
    'graph_type': GraphType.BARABASI_ALBERT,
    'n_vertices': 10,
    'graph_params': {'m': 4},
    'n_episodes': 200,
    'validate_each': 20,
}
experiment_idx = 3
n_runs = 3
agents = train(n_runs, params, PROBLEM, experiment_idx)
