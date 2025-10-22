import numpy as np
import torch
import random
from copy import deepcopy
from argparse import Namespace

from sklearn.externals.array_api_compat.cupy import arange
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

from utils import safe_collate
from reader import *
from model.simulator import *
from env.BaseRLEnvironment import BaseRLEnvironment
import pandas as pd
import math


class KREnvironment_WholeSession_GPU(BaseRLEnvironment):
    '''
    KuaiRand simulated environment for consecutive list-wise recommendation
    Main interface:
    - parse_model_args: for hyperparameters
    - reset: reset online environment, monitor, and corresponding initial observation
    - step: action --> new observation, user feedbacks, and other updated information
    - get_candidate_info: obtain the entire item candidate pool
    Main Components:
    - data reader: self.reader for user profile&history sampler
    - user immediate response model: see self.get_response
    - no user leave model: see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''

    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - uirm_log_path
        - slate_size
        - episode_batch_size
        - item_correlation
        - single_response
        - from BaseRLEnvironment
            - max_step_per_episode
            - initial_temper
        """
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--uirm_log_path', type=str, required=True,
                            help='log path for pretrained user immediate response model')
        parser.add_argument('--slate_size', type=int, required=6,
                            help='number of item per recommendation slate')
        parser.add_argument('--episode_batch_size', type=int, default=32,
                            help='episode sample batch size')
        parser.add_argument('--item_correlation', type=float, default=0,
                            help='magnitude of item correlation')
        parser.add_argument('--single_response', action='store_true',
                            help='only include the first feedback as reward signal')
        return parser

    def __init__(self, args):
        """
        from BaseRLEnvironment:
            self.max_step_per_episode
            self.initial_temper
        self.uirm_log_path
        self.slate_size
        self.rho
        self.immediate_response_stats: reader statistics for user response model
        self.immediate_response_model: the ground truth user response model
        self.max_hist_len
        self.response_types
        self.response_dim: number of feedback_type
        self.response_weights
        self.reader
        self.candidate_iids: [encoded item id]
        self.candidate_item_meta: {'if_{feature_name}': (n_item, feature_dim)}
        self.n_candidate
        self.candidate_item_encoding: (n_item, item_enc_dim)
        self.gt_state_dim: ground truth user state vector dimension
        self.action_dim: slate size
        self.observation_space: see reader.get_statistics()
        self.action_space: n_condidate
        """
        super(KREnvironment_WholeSession_GPU, self).__init__(args)
        self.uirm_log_path = args.uirm_log_path
        self.slate_size = args.slate_size
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation
        self.single_response = args.single_response
        self.window_size = args.window_size

        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline())  # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline())  # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print("Loading raw data")
        assert (class_args.reader == 'KRMBSeqReader' 
                or class_args.reader == 'MLSeqReader'
                or class_args.reader == 'RecKRMBSeqReader'
                ) and 'KRMBUserResponse' in class_args.model

        print(f"mushxx{args}")
        path = f"dataset/{args.dataset}/"
        item_types_np = pd.read_csv(path + 'item_types.csv').to_numpy().astype(np.float32)
        self.item_types = torch.tensor(item_types_np, dtype=torch.float32).to(self.device)

        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args.uirm_log_path, args.dataset)  # definition in base
        self.reader = reader
        print(self.reader.get_statistics())

        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)  # definition in base
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = torch.tensor(list(self.reader.get_response_weights().values())).to(torch.float).to(
            args.device)
        if args.single_response:
            self.response_weights = torch.zeros_like(self.response_weights)
            self.response_weights[0] = 1

        print("Setup candidate item pool")

        # [encoded item id], size (n_item,), [1, 2, ..., num_item + 1]
        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)

        # item meta: {'if_{feature_name}': (n_item, feature_dim)}
        # {item_1:{feature_1:[], feature_2:[]...}, item_2:{feature_1:[], feature_2:[]...}, ...}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        # [n_item, feature_dim]
        for k in candidate_meta[0]:
            self.candidate_item_meta[k] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta])) \
                .view(self.n_candidate, -1).to(self.device)

        # (n_item, item_enc_dim), groud truth encoding is implicit to RL agent
        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids,
                                                                      {k[3:]: v for k, v in
                                                                       self.candidate_item_meta.items()}, 1)
        # [n_item, item_enc_dim]
        self.candidate_item_encoding = item_enc.view(-1, self.immediate_response_model.enc_dim)

        # spaces
        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate

        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device

    def get_candidate_info(self, feed_dict, all_item=True):
        """
        Add entire item pool as candidate for the feed_dict
        @input:
        - all_item: whether obtain all item features from candidate pool
        - feed_dict
        @output:
        - candidate_info: {'item_id': (L,),
                           'if_{feature_name}': (n_item, feature_dim)}
        """
        if all_item:
            candidate_info = {'item_id': self.candidate_iids}
            candidate_info.update(self.candidate_item_meta)
        else:
            candidate_info = {'item_id': feed_dict['item_id']}
            indices = feed_dict['item_id'] - 1
            candidate_info.update({k: v[indices] for k, v in self.candidate_item_meta.items()})
        return candidate_info
    

    def reset(self, params={'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                   'empty_history': True if start from empty history, 
                   'initial_history': start with initial history}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        if 'empty_history' not in params:
            params['empty_history'] = False

        # set inference batch size
        if 'batch_size' in params:
            BS = params['batch_size']
        else:
            BS = self.episode_batch_size

        # random sample users
        self.batch_iter = iter(DataLoader(self.reader, batch_size=BS, shuffle=True,
                                          pin_memory=True, num_workers=4))
        sample_info = next(self.batch_iter)
        self.sample_batch = self.get_observation_from_batch(sample_info)
        self.current_observation = self.sample_batch
        self.current_step = torch.zeros(self.episode_batch_size).to(self.device)
        self.current_sample_head_in_batch = BS

        # user temper for leave model
        self.current_temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.current_pop_temper = torch.ones(self.episode_batch_size).to(self.device) * 5
        self.current_sum_reward = torch.zeros(self.episode_batch_size).to(self.device)

        self.env_history = {'step': [0.], 'leave': [], 'temper': [],
                            'coverage': [], 'coverage_cat': [], 'ILD': []}

        return deepcopy(self.current_observation)

    def step(self, step_dict):
        """
        users react to the recommendation action
        @input:
        - step_dict: {'action': (B, slate_size),
                      'action_features': (B, slate_size, item_dim) }
        @output:
        - new_observation: {'user_profile': {'user_id': (B,),
                                             'uf_{feature_name}': (B, feature_dim)},
                            'user_history': {'history': (B, max_H),
                                             'history_if_{feature_name}': (B, max_H, feature_dim),
                                             'history_{response}': (B, max_H),
                                             'history_length': (B, )}}
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar,
                          'done': (B,)}
        - update_info: see self.update_observation@output - update_info
        """

        # URM forward
        with torch.no_grad():
            action = step_dict['action']  # must be indices on candidate_ids

            # get user response
            response_dict = self.get_response(step_dict)
            response = response_dict['immediate_response']

            # done mask and temper update
            # (B,)
            done_mask = self.get_leave_signal(None, action, response_dict)  # this will also change self.current_temper
            response_dict['done'] = done_mask

            # 3.update user history in current_observation
            # {'slate': (B, slate_size), 'updated_observation': a copy of self.current_observation}
            update_info = self.update_observation(None, action, response, done_mask)

            # 4.env_history update: step, leave, temper, converage, ILD
            self.current_step += 1
            n_leave = done_mask.sum()
            self.env_history['leave'].append(n_leave.item())
            self.env_history['temper'].append(torch.mean(self.current_temper).item())
            self.env_history['coverage'].append(response_dict['coverage'])
            self.env_history['coverage_cat'].append(response_dict['coverage_cat'])

            # ILD: estimates the dissimilarity between items in each recommended list, based on item embedding.
            self.env_history['ILD'].append(response_dict['ILD'])

            # 5.when users left, new users come into the running batch
            org_step = self.current_step.clone()

            if n_leave > 0:
                final_steps = self.current_step[done_mask].detach().cpu().numpy()
                for fst in final_steps:
                    self.env_history['step'].append(fst)

                if self.current_sample_head_in_batch + n_leave < self.episode_batch_size:
                    # reuse previous batch if there are sufficient samples for n_leave
                    head = self.current_sample_head_in_batch
                    tail = self.current_sample_head_in_batch + n_leave
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[head:tail]
                    self.current_sample_head_in_batch += n_leave
                else:
                    # sample new users to fill in the blank
                    sample_info = self.sample_new_batch_from_reader()
                    self.sample_batch = self.get_observation_from_batch(sample_info)
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[:n_leave]
                    self.current_sample_head_in_batch = n_leave
                self.current_step[done_mask] *= 0
                self.current_temper[done_mask] *= 0
                self.current_temper[done_mask] += self.initial_temper
                self.current_pop_temper[done_mask] *= 0
                self.current_pop_temper[done_mask] += self.initial_temper
            else:
                self.env_history['step'].append(self.env_history['step'][-1])
        
        return deepcopy(self.current_observation), response_dict, update_info, org_step

    def get_response(self, step_dict):
        """
        @input:
        - step_dict: {'action': (B, slate_size)}
        @output:
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar}
        """
        # actions (exposures), (B, slate_size), indices of self.candidate_iid
        action = step_dict['action']
        coverage = len(torch.unique(action))
        coverage_cat = len(torch.unique(self.item_types[action]).reshape(-1))
        
        B = self.episode_batch_size

        ########################################
        # This is where the action take effect #
        # (B, action_dim, 1, enc_dim)
        batch = {'item_id': self.candidate_iids[action]}
        batch.update(self.current_observation['user_profile'])
        batch.update(self.current_observation['user_history'])
        batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})
        out_dict = self.immediate_response_model(batch)
        ########################################

        # (B, slate_size, n_feedback)
        behavior_scores = out_dict['probs']

        # (B, slate_size, item_dim)
        item_enc = self.candidate_item_encoding[action].view(B, self.slate_size, -1)
        item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        corr_factor = self.get_intra_slate_similarity(item_enc_norm)

        # user response sampling
        point_scores = behavior_scores - corr_factor.view(B, self.slate_size, 1) * self.rho
        point_scores[point_scores < 0] = 0
        point_scores[point_scores > 1.] = 1.

        response = torch.bernoulli(point_scores).detach()

        point_reward = response * self.response_weights.view(1, 1, -1)
        # (B, slate_size)
        combined_reward = torch.sum(point_reward, dim=2)
        # (B, )
        mean_combined_reward = torch.mean(combined_reward, dim=1)

        return {'immediate_response': response,
                'mean_response': mean_combined_reward,
                'user_state': out_dict['state'],
                # describes the number of distinct items exposed in a mini-batch.
                'coverage': coverage,
                'coverage_cat': coverage_cat,
                # estimates the dissimilarity between items in each recommended list, based on item embedding.
                'ILD': 1 - torch.mean(corr_factor).item()}


    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size, 1, self.gt_state_dim)
        return gt_user_state


    def get_intra_slate_similarity(self, action_item_encoding):
        """
        @input:
        - action_item_encoding: (B, slate_size, enc_dim)
        @output:
        - similarity: (B, slate_size)
        """
        B, L, d = action_item_encoding.shape
        # pairwise similarity in a slate (B, L, L)
        pair_similarity = torch.mean(action_item_encoding.view(B, L, 1, d) * action_item_encoding.view(B, 1, L, d),
                                     dim=-1)
        # similarity to slate average, (B, L)
        point_similarity = torch.mean(pair_similarity, dim=-1)
        return point_similarity

    def get_leave_signal(self, user_state, action, response_dict):
        """
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask: 
        """
        # (B, slate_size, n_feedback)
        point_reward = response_dict['immediate_response'] * self.response_weights.view(1, 1, -1)
        # (B, slate_size)
        combined_reward = torch.sum(point_reward, dim=2)
        # (B, )
        temper_boost = torch.mean(combined_reward, dim=1)

        B = action.shape[0]
        pre_action = (self.current_observation['user_history']['history'])[:, -self.slate_size:].reshape(-1)- 1
        action = action.reshape(-1)
        pre_categories = self.item_types[pre_action].reshape(B, -1)
        cur_categories = self.item_types[action].reshape(B, -1)
        num_equal = (pre_categories == cur_categories).sum(dim=-1).squeeze() * (self.current_step > 0)
        self.current_temper[num_equal > 0] -= 1

        self.current_temper -= 1
        # leave signal
        done_mask = self.current_temper < 1
        return done_mask

    def update_observation(self, user_state, action, response, done_mask, update_current=True):
        """
        user profile stays static, only update user history
        @input:
        - user_state: not used in this env
        - action: (B, slate_size), indices of self.candidate_iids
        - response: (B, slate_size, n_feedback)
        - done_mask: not used in this env
        @output:
        - update_info: {slate: (B, slate_size),
                        updated_observation: same format as self.reset@output - observation}
        """
        # (B, slate_size), convert to encoded item id
        action = action.reshape(action.shape[0], -1)
        rec_list = self.candidate_iids[action]

        # history update
        old_history = self.current_observation['user_history']
        
        item_type = ((self.item_types[action]).float().squeeze() * response[:, :, 0])#.mean(dim=1)
        is_pos_click_num = (item_type.sum(dim=1).long() >= 1).int()#(item_type.sum(dim=1).long() >= 1).int()
        L = old_history['history_length']

        max_H = self.max_hist_len
        L += is_pos_click_num
        #L[L > max_H] = max_H

        new_history = {'history': torch.cat((old_history['history'], rec_list), dim=1)[:, -max_H:], 'history_length': L}

        for k, candidate_meta_features in self.candidate_item_meta.items():
            # (B, slate_size, feature_dim)
            meta_features = candidate_meta_features[action]
            # (B, max_H, feature_dim)
            previous_meta = old_history[f'history_{k}'].view(self.episode_batch_size, max_H, -1)
            new_history[f'history_{k}'] = torch.cat((previous_meta, meta_features), dim=1)[:, -max_H:, :].view(
                self.episode_batch_size, -1)

        # history item responses
        for i, R in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{R}'
            new_history[k] = torch.cat((old_history[k], response[:, :, i]), dim=1)[:, -max_H:]
        if update_current:
            self.current_observation['user_history'] = new_history
        return {'slate': rec_list, 'updated_observation': {
            'user_profile': deepcopy(self.current_observation['user_profile']),
            'user_history': deepcopy(new_history)}}

    def sample_new_batch_from_reader(self):
        """
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        """
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size=self.episode_batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8,))
            sample_info = next(self.batch_iter)
        return sample_info
    
    def reset_new_user(self):
        """
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        """
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size=self.episode_batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8))
            sample_info = next(self.batch_iter)
        

    def stop(self):
        self.batch_iter = None

    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size=B, shuffle=True,
                               pin_memory=True, num_workers=8))

    def create_observation_buffer(self, buffer_size):
        """
        @input:
        - buffer_size: L, scalar
        @output:
        - observation: {'user_profile': {'user_id': (L,),
                                         'uf_{feature_name}': (L, feature_dim)},
                        'user_history': {'history': (L, max_H),
                                         'history_if_{feature_name}': (L, max_H * feature_dim),
                                         'history_{response}': (L, max_H),
                                         'history_length': (L,)}}
        """
        observation = {'user_profile': {'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)},
                       'user_history': {
                           'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device),
                           'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device), 
                           }}
        for f, f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(buffer_size, f_dim).to(torch.float).to(self.device)
        for f, f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(buffer_size, f_dim * self.max_hist_len) \
                .to(torch.float).to(self.device)
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(buffer_size, self.max_hist_len) \
                .to(torch.float).to(self.device)
        return observation

    def get_report(self, smoothness=10):
        return {k: np.mean(v[-smoothness:]) for k, v in self.env_history.items()}


import argparse

if __name__ == '__main__':
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=False, help='Environment class.',
                             default='KREnvironment_WholeSession_GPU')
    initial_args, _ = init_parser.parse_known_args()
    envClass = eval(initial_args.env_class)
    print(envClass)
    print(initial_args)