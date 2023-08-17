import numpy as np
import random
import time
from collections import namedtuple, deque

from model import QNetwork, MPNN
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 16          # minibatch size
GAMMA = 0.10             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR = 5e-2                # learning rate 
CLIP_GRAD_NORM_VALUE = 5 # value of gradient to clip while training
UPDATE_TARGET_EACH = 100 # number of steps to wait until updating target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, nstep, embed_dim=64, T=4, seed=None, gamma=GAMMA,
                 clip_grad_norm_value=CLIP_GRAD_NORM_VALUE):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.nstep = nstep
        if seed is not None:
            random.seed(seed)

        self.gamma = gamma
        self.clip_grad_norm_value = clip_grad_norm_value
        
        # Q-Network
        self.qnetwork_local = QNetwork(embed_dim, T, seed).to(device)
        self.qnetwork_target = QNetwork(embed_dim, T, seed).to(device)
        self.hard_update(self.qnetwork_local, self.qnetwork_target)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.losses = []
        self.update_t_step = 0

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # To be used in n-step learning
        self.discounting_oldest_reward = self.gamma**(self.nstep-1)

        # Initial episode config
        self.reset_episode()

    def reset_episode(self, G=None):
        self.t_step = 0
        self.states = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)
        self.sum_rewards = 0
        # TODO: check if clear_buffer is needed
        self.memory.clear_buffer()
        if G is not None:
            self.qnetwork_local.update_model_info(G)
            self.qnetwork_target.update_model_info(G)

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        reward_to_subtract = self.rewards[0] if self.t_step > self.nstep else 0 # r1

        self.states.append(state[1].copy())
        self.actions.append(action)
        self.rewards.append(reward)

        # Get oldest state and action to add to replay memory buffer
        oldest_state = self.states[0] # s2
        oldest_action = self.actions[0] # a2

        # Get rolling rewards sum
        if self.t_step > self.nstep:
            self.sum_rewards -= reward_to_subtract * self.discounting_oldest_reward
        self.sum_rewards = self.gamma * self.sum_rewards + reward

        # Get xv from info
        if self.t_step >= self.nstep:
            # Save experience in replay memory
            self.memory.add(oldest_state, oldest_action, self.sum_rewards, next_state[1].copy(), done)
            experiences = self.memory.sample()
            self.learn(experiences)
            
        # r2 + G*r1
        # r3 + G*r2 = ((r2 + G*r1) - G*r1)*G + r3
#         nstep = 2
#         t = 3
#         s1 a1 r1 s2 a2 r2 (s3 a3 r3 s4)
        
#         n=1: (s3 a3 r3 s4)
#         n=2: (s2 a2 r2+r3 s4)
#         n=3: (s1 a1 r1+r2+r3 s4)
        
#         (s1, a1, r1 + r2, s3)
#         (s1, a1, r(s1,a1)+r(s2,a2), s3)

    def act(self, obs, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            obs (array_like): current observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy: greedy action selection
        if random.random() < eps:
            return random.choice(valid_actions).item()

        T1 = t1 = time.time()
        state = torch.from_numpy(obs[1]).float().unsqueeze(0).to(device)
        valid_actions = self.get_valid_actions(state)[0]
        t2 = time.time()
        # print('process obs and get valid_actions:', t2-t1)

        t1 = time.time()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = {action: self.qnetwork_local(state, action.unsqueeze(0)).cpu().data.numpy() for action in valid_actions}
        self.qnetwork_local.train()
        t2 = time.time()
        # print('compute action_values:', t2-t1)

        t1 = time.time()
        # Epsilon-greedy: max action-value selection
        ret = max(action_values, key=action_values.get)
        T2 = t2 = time.time()
        # print('compute max action-value:', t2-t1)

        # print('ret', ret, type(ret))
        return ret.item()

    def get_valid_actions(self, states):
        actions = [torch.nonzero(state==0).view(-1) for state in states]
        return [x if x.shape[0] > 0 else torch.tensor([0]) for x in actions]

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute valid actions
        valid_next_states_actions = self.get_valid_actions(next_states)
        
#         print('valid_next_states_actions:', valid_next_states_actions)

#         for valid_next_state_actions in valid_next_states_actions:
#             for valid_next_state_action in valid_next_state_actions:
#                 print('valid_next_state_action:', valid_next_state_action)





#         target_preds = self.qnetwork_target(next_states)
#         disallowed_actions_mask = (next_states[:, 0, :] != self.allowed_action_state)
#         with torch.no_grad():
#             q_targets_next = target_preds.masked_fill(disallowed_actions_mask,-10000).max(1, True)[0]





        # Calc q_targets_next based on current states and valid actions
        # Default value not important, as dones = 1 in this case and Q_targets_next is ignored
        q_targets_next = torch.tensor([
            [max(
                (self.qnetwork_target(next_state.unsqueeze(0), valid_next_state_action.unsqueeze(0)).detach() for valid_next_state_action in valid_next_state_actions),
                default=-1
            )] for (next_state, valid_next_state_actions) in zip(next_states, valid_next_states_actions)
        ]).to(device)
        # TODO: assert q_targets_next.shape = (batch_size, 1)
        # print('q_targets_next.shape:', q_targets_next.shape)

        # Calc q_targets based on Q_targets_next
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

        # Calc q_expected based on current states
        q_expected = self.qnetwork_local(states, actions)

        # Calc loss
        loss = F.mse_loss(q_expected, q_targets)

#         print('Q_hat:', Q_hat)
#         print('Q_targets:', Q_targets)
        
        # Run optimizer step
        self.optimizer.zero_grad()
        # print("theta1 grad:", self.qnetwork_local.theta1.grad)
        self.losses.append(loss.item())
        loss.backward()
        # print("theta1 grad:", self.qnetwork_local.theta1.grad)
        # Gradient clipping to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad_norm_value)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.update_t_step = (self.update_t_step + 1) % UPDATE_TARGET_EACH
        if self.update_t_step == 0:
            self.hard_update(self.qnetwork_local, self.qnetwork_target)
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        
#         # Get max predicted Q values (for next states) from target model
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         # Compute Q targets for current states 
#         Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

#         # Get expected Q values from local model
#         Q_expected = self.qnetwork_local(states).gather(1, actions)

#         # Compute loss
#         loss = F.mse_loss(Q_expected, Q_targets)
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
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

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        target_model.load_state_dict(local_model.state_dict())
