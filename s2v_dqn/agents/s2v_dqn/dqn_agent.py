import numpy as np
import random
from collections import deque

from s2v_dqn.agents.base_agent import BaseAgent
from s2v_dqn.agents.s2v_dqn.model import QNetwork
from s2v_dqn.agents.s2v_dqn.replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)    # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 1.00              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 1e-4                 # learning rate
CLIP_GRAD_NORM_VALUE = 5  # value of gradient to clip while training
UPDATE_TARGET_EACH = 100  # number of steps to wait until updating target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    _SUPPORTED_PROBLEMS = {'mvc', 'tsp'}

    def __init__(self, problem, nstep=1, embedding_dim=64, embedding_layers=4, n_node_features=4,
                 n_edge_features=1, normalize=True, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 gamma=GAMMA, tau=TAU, lr=LR, clip_grad_norm_value=CLIP_GRAD_NORM_VALUE,
                 update_target_each=UPDATE_TARGET_EACH, target_update="hard"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(problem)

        self.nstep = abs(nstep)
        self.use_nstep = nstep > 0

        self.gamma = gamma
        self.clip_grad_norm_value = clip_grad_norm_value
        self.update_target_each = update_target_each
        self.tau = tau
        self.target_update = target_update
        assert target_update in ("soft", "hard"), 'target_update must be one of {"soft", "hard"}'

        # Q-Network
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.qnetwork_local = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                       n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)
        self.qnetwork_target = QNetwork(embed_dim=embedding_dim, embedding_layers=embedding_layers, n_node_features=n_node_features,
                                        n_edge_features=n_edge_features, normalize=normalize).to(device, dtype=torch.float32)

        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta1.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta2.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta3.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.embedding_layer.theta4.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta5.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta6.weight)
        # torch.nn.init.xavier_normal_(self.qnetwork_local.q_layer.theta7.weight)

        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Internal values accross steps 
        self.episode_losses = []
        self.losses = []
        self.q_expecteds = []
        self.q_targets = []
        self.params = []
        # self.grads = []
        self.theta1s = []
        self.theta2s = []
        self.theta3s = []
        self.theta4s = []
        self.theta5s = []
        self.theta6s = []
        self.theta7s = []

        self.update_t_step = 0

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # To be used in n-step learning
        self.gamma_n_minus_1 = self.gamma ** (self.nstep - 1)
        self.gamma_n = self.gamma ** self.nstep

        # Initial episode config
        self.reset_episode()

    def reset_episode(self):
        self.t_step = 0
        self.states = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)
        self.sum_rewards = 0
        # TODO: check if clear_buffer is needed
        # self.memory.clear_buffer()

        if self.qnetwork_local.training:
            self.losses.append(sum(self.episode_losses) / len(self.episode_losses) if len(self.episode_losses) > 0 else 0)
            self.episode_losses = []

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        if not self.use_nstep:
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)
        else:
            reward_to_subtract = self.rewards[0] if self.t_step > self.nstep else 0  # r1

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            # Get oldest state and action to add to replay memory buffer
            oldest_state = self.states[0]  # s2
            oldest_action = self.actions[0]  # a2

            # Get rolling rewards sum
            if self.t_step > self.nstep:
                self.sum_rewards -= reward_to_subtract * self.gamma_n_minus_1
            self.sum_rewards = self.gamma * self.sum_rewards + reward

            # Get xv from info
            if self.t_step >= self.nstep:
                # Save experience in replay memory
                self.memory.add(oldest_state, oldest_action, self.sum_rewards, next_state, done)
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences)

    @torch.no_grad()
    def act(self, obs, *args, **kwargs):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs        : current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        obs = torch.from_numpy(obs).to(device, dtype=torch.float32)
        xv = obs[:, 0]
        valid_actions = (xv == 0).nonzero()

        # IMPORTANT: This should be the first vertex in the tour
        # TODO: replace with env.start_vertex
        if len(valid_actions) == 0:
            return 0

        # Epsilon-greedy: greedy action selection
        eps = kwargs.get("eps", 0.0)
        if random.random() < eps:
            action_idx = np.random.randint(len(valid_actions))
            action = valid_actions[action_idx].item()
            return action

        action_values = self.qnetwork_local(obs).squeeze(0)  # squeeze to remove NN batching
        valid_actions_idx = action_values[valid_actions].argmax().item()
        action = valid_actions[valid_actions_idx].item()
        return action

    # def get_valid_actions(self, states):
    #     actions = [torch.nonzero(state == 0).view(-1) for state in states]
    #     return [x if x.shape[0] > 0 else torch.tensor([0]) for x in actions]

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        target_preds = self.qnetwork_target(next_states)
        invalid_actions_mask = (next_states[:, :, 0] == 1)
        replace = invalid_actions_mask.sum(axis=1) == invalid_actions_mask.shape[1]
        invalid_actions_mask[:, 0] = torch.where(replace, False, invalid_actions_mask[:, 0])

        # Calculate q_targets_next for valid actions
        # target_preds.shape = (batch_size, n_vertices)
        with torch.no_grad():
            q_targets_next = target_preds.masked_fill(invalid_actions_mask, -1e18).max(1, True)[0]

        # Calculate Q value
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calc q_targets based on Q_targets_next
        q_targets = rewards + self.gamma_n * q_targets_next * (1 - dones)

        # Calc loss
        loss = F.mse_loss(q_expected, q_targets)

        self._log_params(actions, dones, invalid_actions_mask, loss, next_states, q_expected, q_targets, states, target_preds)

        # Run optimizer step
        self.optimizer.zero_grad()
        # self.losses.append(loss.item())
        self.episode_losses.append(loss.item())
        loss.backward()
        # self.grads.append(next(self.qnetwork_local.parameters())[0,0].grad)
        # print("theta1 grad:", self.qnetwork_local.embedding_layer.theta1.weight.grad)
        # Gradient clipping to avoid exploding gradient
        if self.clip_grad_norm_value is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.target_update == "soft":
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        elif self.target_update == "hard":
            self.update_t_step = (self.update_t_step + 1) % self.update_target_each
            if self.update_t_step == 0:
                self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def _log_params(self, actions, dones, invalid_actions_mask, loss, next_states, q_expected, q_targets, states, target_preds):
        if loss.item() > 5e150:
            print(f'actions: {list(actions.cpu().detach().numpy().flatten())}')
            print(f'q_expected: {list(q_expected.cpu().detach().numpy().flatten())}')
            print(f'q_targets: {list(q_targets.cpu().detach().numpy().flatten())}')
            print(f'{loss=}')
            print(f'{target_preds=}')
            print(f'{invalid_actions_mask=}')
            print(f'{dones=}')
            print(f'{1 - dones=}')
            print(f'{states=}')
            print(f'{next_states=}')
            print(f'{self.t_step=}')
            import sys
            sys.exit(0)
        self.q_targets.append(q_targets.min().item())
        self.q_expecteds.append(q_expected.min().item())
        # self.params.append(next(self.qnetwork_local.parameters())[0,0].item())
        self.theta1s.append(self.qnetwork_local.embedding_layer.theta1.weight[0, 0].item())
        self.theta2s.append(self.qnetwork_local.embedding_layer.theta2.weight[0, 0].item())
        self.theta3s.append(self.qnetwork_local.embedding_layer.theta3.weight[0, 0].item())
        # self.theta4s.append(self.qnetwork_local.embedding_layer.theta4.weight[0,0].item())
        self.theta5s.append(self.qnetwork_local.q_layer.theta5.weight[0, 0].item())
        self.theta6s.append(self.qnetwork_local.q_layer.theta6.weight[0, 0].item())
        self.theta7s.append(self.qnetwork_local.q_layer.theta7.weight[0, 0].item())

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        target_model.load_state_dict(local_model.state_dict())

    def train(self):
        """Configure PyTorch modules to be in train mode"""
        self.qnetwork_target.train()
        self.qnetwork_local.train()

    def eval(self):
        """Configure PyTorch modules to be in eval mode"""
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()
