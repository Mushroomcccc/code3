import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn
import pandas as pd


class OneStagePolicy_Sac(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - from OneStagePolicy:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        """
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128],
                            help='hidden dim of the action net')
        parser.add_argument('--action_std_init', type=float, default=0.8,
                            help='init action std')
        parser.add_argument('--has_continuous_action_space', type=bool, default=True,
                            help='action type')
        return parser

    def __init__(self, args, environment):
        """
        action_space = {'item_id': ('nominal', stats['n_item']),
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']),
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        """
        super().__init__(args, environment)
        # action is the set of parameters of linear mapping [item_dim, 1]
        self.hyper_action_dim = self.enc_dim + 1
        self.action_dim = self.item_num
        self.effect_action_dim = self.slate_size
        self.has_continuous_action_space = args.has_continuous_action_space
        self.action_layer = nn.Sequential(DNN(self.state_dim, args.policy_action_hidden, self.action_dim,
                                                dropout_rate=self.dropout_rate, do_batch_norm=True), nn.Softmax(dim=-1))
        self.action_std_init = args.action_std_init

    def generate_action(self, state_dict, feed_dict):
        """
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration, set do_effect_action_explore=False in args or do_explore=False in feed_dict
        * Categorical sampling: probabilistic exploration, set do_effect_action_explore=True in args,
                                set do_explore=True and epsilon < 1 in feed_dict
        * Uniform sampling : random exploration, set do_effect_action_explore=True in args,
                             set do_explore=True, epsilon > 0 in feed_dict
        * Gaussian sampling on hyper-action: set do_explore=True, epsilon < 1 in feed_dict
        * Uniform sampling on hyper-action: set do_explore=True, epsilon > 0 in feed_dict

        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as OneStagePolicy.get_forward@input - feed_dict
        @output:
        - out_dict: {'action': (B, K),
                     'action_log_prb': (B, K),
                     'action_prob': (B, K),
                     'reg': scalar}
        """
        state = state_dict['state']
        candidates = feed_dict['candidates']
        epsilon = feed_dict['epsilon']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']

        action_mean = self.action_layer(state)
        dist = Categorical(action_mean)
        action = torch.multinomial(action_mean, num_samples=self.effect_action_dim, replacement=False)
        action_log_prob = torch.mean(dist.log_prob(action.transpose(1, 0)).transpose(1, 0), dim=1).squeeze()

        reg = self.get_regularization(self.action_layer)
        out_dict = {'action': action,
                    'action_log_prob': action_log_prob,
                    'indices': action,
                    'reg': reg}
        return out_dict

    def evaluate(self, feed_dict):
        state = feed_dict['state'].view(-1, self.state_dim)
        action_prob = self.action_layer(state)
        z = (action_prob == 0.0).float() * 1e-8
        log_action_prob = torch.log(action_prob + z)
        return action_prob, log_action_prob

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict
