import datetime
import logging
import os
import re
import sys
from time import time
from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, ReduceLROnPlateau

from s2v_dqn.agents.s2v_dqn.dqn_agent import DQNAgent
from s2v_dqn.envs.base_env import BaseEnv
from s2v_dqn.envs.mvc.mvc_env import MVCEnv
from s2v_dqn.envs.tsp.tsp_env import TSPEnv
from s2v_dqn.instances.instance_generator import InstanceGenerator
from s2v_dqn.utils import plot_graphs


def run_episode(agent: DQNAgent,
                env: BaseEnv,
                eps=0,
                train_mode=True,
                print_actions=False,
                seed=None):
    state, edge_feature = env.reset(seed)
    agent.reset_episode()
    score = 0
    while True:
        action = agent.act(state, edge_feature, eps=eps)
        if print_actions:
            print(action)
        (next_state, next_edge_feature), reward, done = env.step(action)
        score += reward
        if train_mode:
            agent.step(state, edge_feature, action, reward, next_state, next_edge_feature, done)
        state, edge_feature = next_state, next_edge_feature
        if done:
            break
    return score


@torch.inference_mode()
def run_validation(agent: DQNAgent,
                   env: BaseEnv,
                   n_episodes_validation: int,
                   exact_solution_max_size: int = 0,
                   print_to_file=sys.stdout,
                   seed_prefix=None):
    agent.eval()
    val_scores = []
    # isolated = []
    # isolated_in_sol = []
    for val_episode_idx in range(1, n_episodes_validation + 1):
        seed = abs(hash(f"{seed_prefix}_val_{val_episode_idx}")) if seed_prefix is not None else None
        run_episode(agent, env, train_mode=False, seed=seed)
        score = env.get_current_solution()
        if score == 0:
            print(f"{val_episode_idx=} {seed=}")
        solution = env.get_best_solution(exact_solution_max_size)
        if solution == 0:
            print("Solution = 0, seed = ", seed)
            logging.warning("Solution = 0, seed =", seed)
            approximation_ratio = 0
        else:
            approximation_ratio = score / solution # if score >= solution else solution / score
            # print(f"{approximation_ratio=}")
        val_scores.append(approximation_ratio)
        # cnt = 0
        # cnt_in_sol = 0
        # for v in env.graph:
        #     if env.graph.degree[v] == 0:
        #         cnt += 1
        #         if env.xv[v] == 1:
        #             cnt_in_sol += 1
        # isolated.append(cnt)
        # isolated_in_sol.append(cnt_in_sol)
    #         self.covered_edges += np.dot(self.graph_adj_matrix[action], 1 - self.xv)

    val_scores = np.array(val_scores)
    val_stats = pd.Series(val_scores).describe()
    print(
        f"[Validation] "
        f"Mean: {val_stats['mean']}, "
        f"Std dev: {val_stats['std']}, "
        f"Optimal: {100 * (abs(val_scores - 1.0) < 1e-6).sum() / n_episodes_validation :.2f}%",
        # f"Isolated: {sum(isolated_in_sol)/len(isolated_in_sol)} / {sum(isolated)/len(isolated)}",
        file=print_to_file
    )

    agent.train()

    return val_stats


# Function to log weights, gradients, episode reward and optionally the loss
def log_data(writer: Optional[SummaryWriter],
             model: nn.Module,
             episode_idx: int,
             episode_score: float,
             episode_loss: float):
    # Skip logging if it's not configured
    if writer is None:
        return

    # Log weights and gradients
    for name, param in model.named_parameters():
        #         print(f"{name=}, {param=}, {episode=}")
        writer.add_histogram(f"weights/{name}", param, episode_idx)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, episode_idx)

    # Log the episode reward
    writer.add_scalar('Episode Reward', episode_score, episode_idx)
    # Optionally log the loss
    if episode_loss is not None:
        writer.add_scalar('Loss', episode_loss, episode_idx)


def run_train(problem: str,
              agent: DQNAgent,
              env: BaseEnv,
              eps_start: float,
              eps_decay: float,
              eps_end: float,
              scheduler: Union[LRScheduler, Callable] = None,
              n_episodes: int = 10000,
              start_episode: int = 1,
              validate_each: int = 25,
              n_episodes_validation: int = 10,
              print_train_metrics_each: int = 100,
              print_thetas: bool = True,
              validate_at_start: bool = True,
              print_to_file=sys.stdout,
              experiment_idx: int = None,
              run_idx: int = None,
              tensorboard_log: bool = False,
              exact_solution_max_size: int = 0):
    problem = problem.lower()

    # Set up TensorBoard logging
    if tensorboard_log:
        if experiment_idx is not None:
            log_file = "exp_{}{}".format(experiment_idx, f"_{run_idx}" if run_idx is not None else "")
        else:
            log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(f"..", "logs", problem, "fit", log_file)
        print(f"{log_path=}")
        writer = SummaryWriter(log_path)
    else:
        writer = None

    scores = []
    val_scores = []
    eps = eps_start
    start_time = time()
    if validate_at_start:
        seed_prefix = f"{problem}_{experiment_idx}_{run_idx}_0"
        val_score = run_validation(
            agent,
            env,
            n_episodes_validation,
            print_to_file=print_to_file,
            exact_solution_max_size=exact_solution_max_size,
            seed_prefix=seed_prefix,
        )['mean']
        val_scores.append(val_score)
    for episode_idx in range(start_episode, n_episodes + 1):
        try:
            # run episode for given episode_idx
            seed = abs(hash(f"{problem}_{experiment_idx}_{run_idx}_{episode_idx}_train"))
            episode_score = run_episode(agent, env, eps, train_mode=True, seed=seed)
            episode_loss = agent.losses[-1] if len(agent.losses) > 0 else None
            eps = max(eps * eps_decay if eps_decay >= 0 else eps + eps_decay, eps_end)
            scores.append(episode_score)

            # update the LR scheduler
            if isinstance(scheduler, LambdaLR):
                scheduler.step()
            elif isinstance(scheduler, Callable):
                lr = scheduler(episode_idx)
                for g in agent.optimizer.param_groups:
                    g['lr'] = lr

            if episode_idx % print_train_metrics_each == 0 and len(agent.q_targets) > 0:
                print(
                    f"[{episode_idx}/{n_episodes}] "
                    f"loss: {agent.losses[-1]:.3e}, "
                    f"q_target: {agent.q_targets[-1]:.3e}, "
                    f"q_expected: {agent.q_expecteds[-1]:.3e}, "
                    f"eps: {eps:.4f}, "
                    f"time: {time() - start_time:.2f}s",
                    file=print_to_file,
                    flush=True
                )
                if print_thetas and len(agent.theta1s) > 0:
                    print(f"    θ1: {agent.theta1s[-1]:.3e}, "
                          f"θ2: {agent.theta2s[-1]:.3e}, "
                          f"θ3: {agent.theta3s[-1]:.3e}, "
                          # f"θ4: {agent.theta4s[-1]:.3e}, "
                          f"θ5: {agent.theta5s[-1]:.3e}, "
                          f"θ6: {agent.theta6s[-1]:.3e}, "
                          f"θ7: {agent.theta7s[-1]:.3e}",
                          file=print_to_file
                          )

            # check if we should run validation
            if episode_idx % validate_each == 0:
                seed_prefix = f"{problem}_{experiment_idx}_{run_idx}_{episode_idx}"
                val_score = run_validation(
                    agent,
                    env,
                    n_episodes_validation,
                    print_to_file=print_to_file,
                    exact_solution_max_size=exact_solution_max_size,
                    seed_prefix=seed_prefix,
                )['mean']
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_score)
                val_scores.append(val_score)
                agent.save_model(f"../checkpoints/{problem}_{experiment_idx}_{run_idx}_{episode_idx}.pth")

            # Log weights, gradients, episode reward and optionally the loss
            log_data(writer, agent.qnetwork_local, episode_idx, episode_score, episode_loss)

        except KeyboardInterrupt:
            print(f"Training interrupted in episode #{episode_idx}")
            break
    print(f"Total time: {time() - start_time:.2f}s", file=print_to_file)

    # Close Tensorboard writer
    if writer:
        writer.close()

    return scores, val_scores


def train(n_runs: int,
          params: dict,
          problem: str,
          experiment_idx: Optional[int] = None):

    problem = problem.lower()
    assert problem in {'mvc', 'tsp'}

    # Setup filename correctly
    filename_pattern = 'outputs-{}/run_{}.{}'.format(problem, '{}', '{}')

    # Setup experiment_idx based on already existing experiments in given folder
    if experiment_idx is None:
        folder = path_parts[0] if len(path_parts := os.path.split(filename_pattern)) > 1 else './'

        indices = set(int(match[0]) for file in os.listdir(folder) if (match := re.findall(r'run_(\d+)\.log', file)) != [])
        experiment_idx = min(i for i in range(len(indices) + 1) if i not in indices)
        print('Experiment idx:', experiment_idx)

    # Params with defaults
    decay_type = params.get('decay_type', 'linear')
    validate_at_start = params.get('validate_at_start', False)
    env_extra_params = params.get('env_extra_params', {})
    exact_solution_max_size = params.get('exact_solution_max_size', 20)
    discount_factor = params.get('discount_factor', 1.0)
    update_params_each = params.get('update_params_each', 4)
    warmup_steps = params.get('warmup_steps', 1000)
    target_update = params.get('target_update', 'hard')
    double_dqn = params.get('double_dqn', False)
    tau = params.get('tau', 5e-3)
    update_target_each = params.get('update_target_each', 500)
    tensorboard_log = params.get('tensorboard_log', False)

    # Params without defaults
    n_vertices = params['n_vertices']
    embedding_layers = params['embedding_layers']
    n_episodes = params['n_episodes']
    eps_start = params['eps_start']
    eps_end = params['eps_end']
    eps_end_at_episode = params['eps_end_at_episode']
    batch_size = params['batch_size']
    graph_params = params['graph_params']

    # transform relative to absolute episode number
    lr_config = params['lr_config']
    lr_config = [(int(ep * n_episodes), lr) if 0 < ep <= 1 else (ep, lr) for (ep, lr) in lr_config]

    f_log = open(filename_pattern.format(experiment_idx, 'log'), 'w')

    print({"problem": problem, **params})
    print({"problem": problem, **params}, file=f_log)

    instance_generator = InstanceGenerator(
        n_min=n_vertices,
        n_max=n_vertices,
        graph_type=params['graph_type'],
        graph_params=graph_params
    )

    env_class = {
        'mvc': MVCEnv,
        'tsp': TSPEnv,
    }[problem]
    env = env_class(instance_generator, **env_extra_params)

    all_agent_losses = []
    all_val_scores = []
    agents = []
    for run_idx in range(n_runs):
        agent = DQNAgent(
            problem,
            nstep=params.get('nstep', 1),
            normalize=params.get('normalize', True),
            batch_size=batch_size,
            lr=lr_config[0][1],
            n_node_features=env.n_node_features,
            n_edge_features=env.n_edge_features,
            embedding_layers=embedding_layers,
            gamma=discount_factor,
            update_params_each=update_params_each,
            warmup_steps=warmup_steps,
            target_update=target_update,
            double_dqn=double_dqn,
            tau=tau,
            update_target_each=update_target_each
        )

        def lr_lambda(_lr_config: list):
            """
            lr_config: list of tuples (start_episode, lr_value)
            """
            def f(episode):
                for i in range(len(_lr_config) - 1, -1, -1):
                    start_episode, lr_value = _lr_config[i]
                    if episode >= start_episode:
                        return lr_value
                raise ValueError("lr_config should have its first element starting at episode 0")
            return f

        # scheduler = LambdaLR(agent.optimizer, lr_lambda=lr_lambda(lr_config))
        scheduler = lr_lambda(lr_config)

        if 0 < eps_end_at_episode <= 1:
            eps_end_at_episode = round(eps_end_at_episode * n_episodes)

        if decay_type == 'exponential':
            # exponential decay derived from formula eps_end = eps_start * (eps_decay ** episode)
            eps_decay = (eps_end / eps_start) ** (1 / eps_end_at_episode)
        elif decay_type == 'linear':
            # linear decay derived from formula eps_end = eps_start + (eps_decay * episode)
            eps_decay = (eps_end - eps_start) / eps_end_at_episode
        else:
            raise ValueError(f"decay_type should be exponential or linear, but found {decay_type}")

        print(f'Starting run #{run_idx+1}/{n_runs}...')
        print(f'Starting run #{run_idx+1}/{n_runs}...', file=f_log)
        scores, val_scores = run_train(
            problem,
            agent,
            env,
            eps_start,
            eps_decay,
            eps_end,
            scheduler=scheduler,
            n_episodes=n_episodes,
            validate_each=params['validate_each'],
            print_train_metrics_each=params['print_train_metrics_each'],
            print_thetas=False,
            print_to_file=f_log,
            validate_at_start=validate_at_start,
            experiment_idx=experiment_idx,
            run_idx=run_idx,
            exact_solution_max_size=exact_solution_max_size,
            tensorboard_log=tensorboard_log,
        )

        agents.append(agent)
        # print(f"{len(agent.losses)=}")
        # print(type(val_scores), len(val_scores))
        all_agent_losses.append(agent.losses)
        all_val_scores.append(val_scores)

    with open(filename_pattern.format(f'{experiment_idx}_loss', 'log'), 'w') as f_loss:
        print(all_agent_losses, file=f_loss)
    with open(filename_pattern.format(f'{experiment_idx}_val', 'log'), 'w') as f_val:
        print(all_val_scores, file=f_val)

    f_log.close()

    # print(filename.format(experiment_idx, 'png'))
    filename_plot = filename_pattern.format(experiment_idx, 'png')
    plot_graphs(
        all_agent_losses,
        all_val_scores,
        max_loss=0.1,
        save_to_path=filename_plot,
        n_vertices=n_vertices,
        lr_config=lr_config,
        batch_size=batch_size,
        validate_each=params['validate_each'],
        validate_at_start=validate_at_start,
        problem=problem,
        graph_type=params['graph_type'],
    )

    return agents
