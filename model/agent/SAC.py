import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent
import pandas as pd
    
class SAC(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        """
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='mitigation factor')
        return parser

    def __init__(self, *input_args):
        """
        components:
        - critic
        - critic_optimizer
        - critic_target
        - components from BaseRLAgent:
            - env
            - actor
            - actor_optimizer
            - buffer
            - exploration_scheduler
            - registered_models
        """
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        path = f"dataset/{args.dataset}/"
        self.item_popularity = torch.tensor(pd.read_csv(path + 'item_popularity.csv').to_numpy()).to(self.device)

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.gamma_n = args.gamma
        self.avg_prob = 1.0 / self.actor.action_dim
        self.maximum_entropy = -np.log(self.avg_prob) * 0.8

        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.tensor([-1.5], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # controller
        self.critic_optimizer1 = torch.optim.Adam(self.critic.net1.parameters(), lr=args.critic_lr,
                                                 weight_decay=args.critic_decay)
        self.critic_optimizer2 = torch.optim.Adam(self.critic.net2.parameters(), lr=args.critic_lr,
                                                 weight_decay=args.critic_decay)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.actor_lr)

        self.do_actor_update = True
        self.do_critic_update = True
        # register models that will be saved
        self.registered_models.append((self.critic, self.critic_optimizer1, "_critic1"))
        self.registered_models.append((self.critic, self.critic_optimizer2, "_critic2"))

    def setup_monitors(self):
        """
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        """
        super().setup_monitors()
        self.training_history.update({'actor_loss': [], 'critic_loss': [], 'Q1': [], 'Q2': [], 
                                      'next_Q': [], 'next_V': [],'entroy_loss': []})

    def step_train(self):
        """
        @process:
        - get sample
        - calculate Q'(s_{t+1}, a_{t+1}) and Q(s_t, a_t)
        - critic loss: TD error loss
        - critic optimization
        - actor loss: Q(s_t, \pi(s_t)) maximization
        - actor optimization
        """

        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        epsilon = 0
        is_train = True
        # (B, )
        reward = user_feedback['reward'].view(-1)
        item_id = policy_output['action']
        action = policy_output['action']
        popularity = self.item_popularity[item_id].reshape(reward.shape[0], -1).reshape(reward.shape[0], -1)
        reward_pop = 1 / (torch.log(popularity.mean(dim=1) + 1.1) + 1)
        # print(reward, reward_pop * 0.01)
        reward += reward_pop * 0.1
        
        # 
        self.alpha = self.log_alpha.exp().detach()
        print(self.alpha)
        with torch.no_grad():
            next_policy_output = self.apply_policy(next_observation, self.actor, 0., False, is_train)
            action_probs, log_action_probs = self.actor.evaluate(next_policy_output)
            next_q = self.apply_critic(next_observation, next_policy_output, self.critic_target)['q']
            next_v = (action_probs * (next_q - self.alpha * log_action_probs)).sum(dim=1)
            target_q = reward + next_v * (~done_mask) * self.gamma_n
        critic_out = self.apply_critic(observation, policy_output, self.critic)
        curr_q1, curr_q2 = critic_out['q1'], critic_out['q2']
        curr_q1 = torch.mean(curr_q1.gather(1, action.long()), dim=1)
        curr_q2 = torch.mean(curr_q2.gather(1, action.long()), dim=1)
        q1_loss = torch.mean((curr_q1 - target_q.detach()).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q.detach()).pow(2))
        self.update_params(self.critic_optimizer1, q1_loss)
        self.update_params(self.critic_optimizer2, q2_loss)

        action_probs, log_action_probs = self.actor.evaluate(policy_output)
        with torch.no_grad():
            q = self.apply_critic(observation, policy_output, self.critic)['q'].detach()
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
        v = torch.sum(action_probs * q, dim=1, keepdim=True)
        policy_loss = -(self.alpha * entropies + v).mean()
        self.update_params(self.actor_optimizer, policy_loss)

        # 计算熵损失
        entropy_loss = torch.mean(self.log_alpha * (self.maximum_entropy - entropies.detach()))
        self.update_params(self.alpha_optimizer, entropy_loss)

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict = {'actor_loss': policy_loss.item(),
                     'critic_loss': q1_loss.item(),
                     'Q1': torch.mean(curr_q1).item(),
                     'Q2': torch.mean(curr_q2).item(),
                     'next_Q': torch.mean(next_q).item(),
                     'next_V': torch.mean(next_v).item(),
                     'entroy_loss': torch.mean(entropy_loss).item()}

        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])

        return loss_dict

    def apply_policy(self, observation, actor, *policy_args):
        """
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        """
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation,
                      'candidates': self.env.get_candidate_info(observation),
                      'epsilon': epsilon,
                      'do_explore': do_explore,
                      'is_train': is_train,
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict

    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state']}
        return critic(feed_dict)
    
    def update_params(self, optim, loss, retain_graph=True):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
