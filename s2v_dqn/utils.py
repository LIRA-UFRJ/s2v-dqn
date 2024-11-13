import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchviz import make_dot

from s2v_dqn.graph_type import GraphType


def from_project_root(path) -> str:
    root_path = Path(__file__).absolute().parent.parent
    return str(root_path.joinpath(path))


def plot_graphs(agent_losses, val_scores, max_loss=1.0, save_to_path=None, **kwargs):
    agent_losses = np.atleast_2d(agent_losses)
    val_scores = np.atleast_2d(val_scores)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), facecolor='white')

    losses_df = pd.DataFrame(agent_losses).T
    first_non_zero_idx = losses_df.ne(0).prod(axis=1).idxmax()
    losses_df = losses_df.iloc[first_non_zero_idx:, :]
    losses_x = pd.DataFrame({'x': np.arange(len(agent_losses[0]))})

    # print(f"{losses_df=}")
    # print(f"{cat.head(20)=}")
    # print(f"{cat.tail(20)=}")

    # pd.concat(
    #     [losses_df, losses_x],
    #     axis=1
    # )
    losses_df.plot(
        use_index=True,
        # x='x',
        color='b',
        alpha=0.05,
        legend=False,
        xlabel='Episode',
        ylabel='Loss Value',
        ax=ax[0]
    )
    losses_df_mean = losses_df.mean(axis=1)
    losses_df_mean.plot(color='r', title=f'Loss Function', ax=ax[0])

    losses_df_stddev = losses_df.std(axis=1)
    losses_df_lower = losses_df_mean - losses_df_stddev
    losses_df_upper = losses_df_mean + losses_df_stddev

    # print(f"{losses_x.shape=}")
    # print(f"{losses_df.shape=}")
    # print(f"{losses_df_lower.shape=}")
    # print(f"{losses_df_upper.shape=}")

    ax[0].fill_between(
        np.arange(first_non_zero_idx, first_non_zero_idx + len(losses_df_lower)),
        losses_df_lower,
        losses_df_upper,
        facecolor='b',
        alpha=0.2
    )

    ax[0].set_xlabel('Episode')
    ax[0].set_ylim((0, max_loss))
    ax[0].set_title(f'Loss function')

    val_df = pd.DataFrame(val_scores).T
    val_x_start = 0 if kwargs.get('validate_at_start', False) else 1
    val_x = pd.DataFrame({'x': np.arange(val_x_start, len(val_scores[0])) * kwargs.get('validate_each', 1)})

    pd.concat(
        [val_df, val_x],
        axis=1
    ).plot(
        x='x',
        color='b',
        alpha=0.05,
        legend=False,
        xlabel='Episode',
        ylabel='Approximation Ratio',
        ax=ax[1]
    )
    val_df_mean = val_df.mean(axis=1)
    val_df_stddev = val_df.std(axis=1)
    val_df_lower = val_df_mean - val_df_stddev
    val_df_upper = val_df_mean + val_df_stddev

    # print(f"{val_x=}")
    # print(f"{val_df_mean=}")
    # print(f"{val_df_lower=}")
    # print(f"{val_df_upper=}")

    pd.concat([val_df_mean, val_x], axis=1).plot(x='x', color='r', legend=False, title=f'Validation scores', ax=ax[1])
    ax[1].fill_between(val_x.to_numpy().flatten(), val_df_lower, val_df_upper, facecolor='b', alpha=0.2)
    ax[1].set_xlabel('Episode')
    ax[1].set_title('Validation scores')

    print(f'Min of avg validation score across episodes: {val_df_mean.min()}')

    # subtitle_keys = ['n_vertices', 'lr_config', 'batch_size']
    subtitle_keys = ['n_vertices', 'batch_size']
    subtitle = ', '.join([f'{k} = {kwargs[k]}' for k in subtitle_keys])
    title = f"{kwargs['problem'].upper()} - {GraphType(kwargs['graph_type']).name}"

    fig.suptitle(f'{title}\n{subtitle}')
    plt.tight_layout()

    if save_to_path is not None:
        plt.savefig(save_to_path, facecolor='white', transparent=False)
    plt.show()


def replay_graphs(problem: str, experiment_idx: int, max_loss=1.0, save_img=False):
    problem = problem.lower()

    # Setup filename correctly
    filename_pattern = 'outputs-{}/run_{}{}.{}'.format(problem, experiment_idx, '{}', '{}')
    save_to_path = filename_pattern.format('', 'png') if save_img else None

    with open(filename_pattern.format('_loss', 'log')) as f:
        agent_losses_str = f.read()
    agent_losses = ast.literal_eval(agent_losses_str)
    with open(filename_pattern.format('_val', 'log')) as f:
        val_scores_str = f.read()
    with open(filename_pattern.format('', 'log')) as f:
        config_str = f.readlines()[0].strip()
    # config_parsed = re.sub(r'<([^\s]+):\s[^>]+>', r'\1', config_str)
    config_parsed = re.sub(r"('graph_type'): <[^\s]+:\s([^>]+)>", r"\1: \2", config_str)

    print(f"{config_parsed=}")
    config = ast.literal_eval(config_parsed)
    val_scores = np.array(ast.literal_eval(val_scores_str))

    plot_graphs(
        agent_losses,
        val_scores,
        max_loss=max_loss,
        save_to_path=save_to_path,
        n_vertices=config['n_vertices'],
        lr_config=config['lr_config'],
        batch_size=config['batch_size'],
        validate_each=config.get('validate_each', 1),
        validate_at_start=config.get('validate_at_start', False),
        problem=config['problem'],
        graph_type=config['graph_type'],
    )


def visualize_pytorch_graph(module, *args):
    module_name = module._get_name()
    make_dot(
        module(*args),
        params=dict(module.named_parameters())
    ).render(f"torchviz_{module_name}", format="png")
    sys.exit(0)
