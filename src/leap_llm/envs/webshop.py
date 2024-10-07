# Local copy from https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/env.py

import sys
import json
import random
import argparse
from os.path import join, dirname, abspath
from collections import defaultdict

# connect to WebShop env
MODEL_PATH = dirname(abspath(__file__))
SITE_PATH = join(MODEL_PATH, '../')
sys.path.insert(0, SITE_PATH)

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import *
from web_agent_site.engine.goal import get_reward

def parse_args():
    parser = argparse.ArgumentParser()
    # logging
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--ckpt_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--test_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--wandb', default=1, type=int)

    # rl
    parser.add_argument('--num_envs', default=4, type=int)
    parser.add_argument('--step_limit', default=100, type=int)
    parser.add_argument('--max_steps', default=300000, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--clip', default=10, type=float)
    parser.add_argument('--bptt', default=8, type=int)
    parser.add_argument('--exploration_method', default='softmax', type=str, choices=['eps', 'softmax'])
    parser.add_argument('--w_pg', default=1, type=float)
    parser.add_argument('--w_td', default=1, type=float)
    parser.add_argument('--w_il', default=0, type=float)
    parser.add_argument('--w_en', default=1, type=float)

    # model
    parser.add_argument('--network', default='bert', type=str, choices=['bert', 'rnn'])
    parser.add_argument('--bert_path', default="", type=str, help='which bert to load')
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--grad_encoder', default=1, type=int)
    parser.add_argument('--get_image', default=0, type=int, help='use image in models')

    # env
    parser.add_argument('--num', default=None, type=int)
    parser.add_argument('--click_item_name', default=1, type=int)
    parser.add_argument('--state_format', default='text_rich', type=str)
    parser.add_argument('--human_goals', default=1, type=int, help='use human goals')
    parser.add_argument('--num_prev_obs', default=0, type=int, help='number of previous observations')
    parser.add_argument('--num_prev_actions', default=0, type=int, help='number of previous actions')
    parser.add_argument('--extra_search_path', default="", type=str, help='path for extra search queries')
    

    # experimental 
    parser.add_argument('--ban_buy', default=0, type=int, help='ban buy action before selecting options')
    parser.add_argument('--score_handicap', default=0, type=int, help='provide score in state')
    parser.add_argument('--go_to_item', default=0, type=int)
    parser.add_argument('--go_to_search', default=0, type=int)
    parser.add_argument('--harsh_reward', default=0, type=int)


    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    return parser.parse_known_args()


class WebEnv:
    ''' A wrapper of textEnv for models. Returns valid actions at each step of the game. '''

    def __init__(self, args, split, server=None, id=None):
        self.env = WebAgentTextEnv(observation_mode=args.state_format, server=server,
                                   filter_goals=None, limit_goals=-1,
                                   num_products=args.num, human_goals=args.human_goals,
                                   get_image=args.get_image,
                                   num_prev_obs=args.num_prev_obs, num_prev_actions=args.num_prev_actions,
                                   session_prefix=id)
        if args.num is None:
            if split == 'test':
                self.goal_idxs = range(500)
            elif split == 'eval':
                self.goal_idxs = range(500, 1500)
            elif split == 'train':
                self.goal_idxs = range(1500, len(self.env.server.goals))
        else:
            self.goal_idxs = range(len(self.env.server.goals))
            
        self.steps = 0
        self.step_limit = args.step_limit
        self.stats = defaultdict(int)  # kept across episodes
        self.session = None
        self.click_item_name = args.click_item_name
        self.asin2name = {k.lower(): v['Title'].lower(
        ) for k, v in self.env.server.product_item_dict.items()}
        self.name2asin = {v: k for k, v in self.asin2name.items()}
        self.attributes_fail = defaultdict(int)
        self.attributes_success = defaultdict(int)
        self.items_clicked = defaultdict(int)
        self.harsh_reward = args.harsh_reward
        self.go_to_item = args.go_to_item
        self.go_to_search = args.go_to_search
        self.ban_buy = args.ban_buy
        self.prev_ob = self.cur_ob = None
        self.get_image = args.get_image
        self.item_rank = -1
        self.reduce_click = 1

        if args.extra_search_path != "":
            self.extra_search = json.load(open(args.extra_search_path))
            self.extra_search = {
                k.strip("."): v for k, v in self.extra_search.items()}
        else:
            self.extra_search = None

    def get_search_texts(self, atts, query, inst):
        # TODO: make it more complicated, or replace it with free-form generation
        if self.extra_search is not None:
            if ", and price lower than" in inst:
                idx = inst.find(", and price lower than")
                inst_ = inst[:idx]
            else:
                inst_ = inst
            texts = self.extra_search.get(inst_, []) + [inst.lower()]
        else:
            texts = [query] + \
                [f'{att} {query}' for att in atts] + [inst.lower()]
        return texts

    def get_valid_actions(self):
        valid_info = self.env.get_available_actions()
        if valid_info['has_search_bar']:  # only search action available
            atts = self.session['goal']['attributes']
            query = self.session['goal']['query']
            inst = self.session['goal']['instruction_text']
            texts = self.get_search_texts(atts, query, inst)
            #valids = [f'search[{text}]' for text in texts]
            valids = [f'search[<search query>]']
        else:
            valids = []  # and text.startswith('b')]
            for text in valid_info['clickables']:
                # ban buy when options not completed
                if text == 'buy now' and self.ban_buy:
                    cur_options = len(self.session['options'])
                    all_options = len(
                        self.env.server.product_item_dict[self.session["asin"]]['customization_options'])
                    if cur_options != all_options:
                        continue
                if text != 'search':
                    if self.click_item_name and text in self.asin2name:
                        text = 'item - ' + self.asin2name[text]
                    valids.append(f'click[{text}]')
                # do some action space reduction...
                if self.reduce_click and len(valids) > 20:
                    valids = valids[:6] + random.sample(valids[6:], 10)
        if len(valids) == 0:
            valids = ['finish']
        return valids

    def score(self):
        """
        Calculate the score of the current state.
        """
        valid_acts = self.get_valid_actions()
        if 'click[description]' not in valid_acts:
            return 0.0
        product = self.env.server.product_item_dict[self.session["asin"]]
        goal = self.session['goal']
        price = self.env.server.product_prices.get(self.session["asin"])
        options = self.session['options']
        return get_reward(product, goal, price, options)

    def estimate_score(self, atts, opts, verify=False):
        """
        Calculate the score of the current state.
        """
        valid_acts = self.get_valid_actions()
        assert 'click[description]' in valid_acts
        # estimate r_att
        desc = self.step('click[description]')[0].lower()
        self.step('click[< prev]')
        feat = self.step('click[features]')[0].lower()
        ob = self.step('click[< prev]')[0].lower()
        n_att = 0
        for att in atts:
            if att in desc or att in feat or att in ob:
                n_att += 1
        r_att = n_att / len(atts)
        # estimate r_opt
        n_opt = 0
        for opt in opts:
            for act in valid_acts:
                if opt in act:
                    n_opt += 1
                    break
        r_opt = n_opt / len(opts)

        r = (n_att + n_opt + 1) / (len(atts) + len(opts) + 1)
        return r, r_att, r_opt

    def step(self, action):
        if self.click_item_name and action.startswith('click[item - ') and action[13:-1] in self.name2asin:
            valid_items = [_ for _ in self.get_valid_actions()
                           if _.startswith('click[item - ')]
            if action in valid_items:
                self.item_rank = valid_items.index(action) + 1
            else:
                self.item_rank = -1
            action = f'click[{self.name2asin[action[13:-1]]}]'

        ob, reward, done, info = self.env.step(action)

        if action.startswith('click[') and action[6:-1] in self.asin2name:
            self.items_clicked[action[6:-1]] += 1
            desc = self.env.step('click[description]')[0].lower()
            self.env.step('click[< prev]')
            feat = self.env.step('click[features]')[0].lower()
            self.env.step('click[< prev]')
        else:
            desc = feat = ''
        r_visit = 0.0
        self.cur_ob, self.prev_ob = ob, self.cur_ob
        if info is None:
            info = {}
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        if done:
            info['verbose'] = self.session.get('verbose_info', {
                                               'r_att': 0.0, 'r_option': 0.0, 'r_price': 0.0, 'r_type': 0.0, 'w_att': 0.0, 'w_option': 0.0, 'w_price': 0.0})
            verbose = info['verbose']
            verbose['r_harsh'] = (reward == 1)
            verbose['r_exact'] = (reward == 1) and (
                self.session['goal']['asin'] == self.session['asin'])
            verbose['r_norm'] = reward / self.steps
            verbose['r_visit'] = r_visit
            verbose['rank_item'] = self.item_rank
            # log reward with respect to #options
            if self.harsh_reward:
                reward = verbose['r_harsh']
            for k, v in self.session['actions'].items():
                self.stats[f'action_{k}'] += v
            cat = self.session['goal']['category']
            self.stats[f'cat_{cat}'] += 1
            for att in self.session['goal']['attributes']:
                if att in info['verbose'].get('purchased_attrs', []):
                    self.attributes_success[att] += 1
                else:
                    self.attributes_fail[att] += 1

        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text,
                     'score': reward * 10, 'estimate_score': self.score(),
                     'prev_ob': self.prev_ob, 'desc': desc, 'feat': feat
                     })

        if self.get_image:
            image_feat = self.env.get_image()
            info['image_feat'] = image_feat

        return ob, (reward + r_visit) * 10, done, info

    def reset(self, idx=None):
        if idx is None:
            idx = random.sample(self.goal_idxs, k=1)[0]
        ob, info = self.env.reset(idx)
        self.session = self.env.server.user_sessions[self.env.session]
        if info is None:
            info = {}
        self.cur_ob, self.prev_ob = ob, None
        info.update({'valid': self.get_valid_actions(), 'goal': self.env.instruction_text,
                     'score': 0, 'estimate_score': self.score(),
                     'prev_ob': self.prev_ob, 'desc': '', 'feat': ''
                     })
        self.steps = 0
        if self.go_to_search or self.go_to_item:
            name = self.session['goal']['name'].lower()
            ob, _, _, info = self.step(f'search[{name}]')
            self.stats['action_go_to_search'] += 1
            if self.go_to_item:
                asin = self.session['goal']['asin'].lower()
                if asin in self.env.get_available_actions()['clickables']:
                    ob, _, _, info = self.step(f'click[{asin}]')
                    self.stats['action_go_to_item'] += 1

        self.item_rank = -1
        return ob, info

    def close(self):
        self.env.close()
