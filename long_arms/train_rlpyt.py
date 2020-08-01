##############################################################################
# Figuring out rlpyt
#
# Some documentations:
#   Runner: https://rlpyt.readthedocs.io/en/latest/pages/runner.html
#   Sampler: https://rlpyt.readthedocs.io/en/latest/pages/sampler.html
#
# Algorithms
# Base github: https://github.com/astooke/rlpyt/blob/master/
#   agent: rlpyt/agents/dqn/catdqn_agent.py
#   model: rlpyt/models/dqn/atari_catdqn_model.py
#   algot: rlpyt/algos/dqn/cat_dqn.py
##############################################################################

import argparse
import configparser
import math
import random  # only imported to set seed
import sys

import gym
import numpy as np
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from r0d1.algo_r0d1 import R0D1
from envs.long_arms import LongArmsEnv
from envs.logical_arms import LogicalArmsEnv


def env_f(**kwargs):
    return GymEnvWrapper(LogicalArmsEnv(**kwargs))


def build_and_train(config: configparser.ConfigParser,
                    cuda_idx: int = None,
                    n_parallel: int = 1,
                    log_dir: str = None):
    # =====
    # Set up hardware
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")

    # =====
    # Set up environment
    img_len = config['Env'].getint('img_len')  # side length of the cifar img
    corridor_len = config['Env'].getint('corridor_length')
    env_args = {
        'num_arms': config['Env'].getint('num_arms'),
        'corridor_length': corridor_len,
        'final_obs_aliased': config['Env'].getboolean('final_obs_aliased'),
        'require_final_action': config['Env'].getboolean('require_final_action'),
        'img_size': (img_len, img_len),
        'grayscale': config['Env'].getboolean('grayscale'),
        'flatten_obs': config['Env'].getboolean('flatten_obs'),
        'scale_observation': config['Env'].getboolean('scale_observation'),
        'dataset_path': config['Env']['dataset_path']
    }
    train_env_args = {k: env_args[k] for k in env_args}  # 1-depth copying
    train_env_args['training'] = True
    eval_env_args = {k: env_args[k] for k in env_args}
    eval_env_args['training'] = False  # NOTE: so far these are just dummies

    # =====
    # Set up algorithm
    algo_kwargs = {
        'discount': config['Algorithm'].getfloat('discount'),
        'lambda_coef': config['Algorithm'].getfloat('lambda_coef'),
        'use_recurrence': config['Algorithm'].getboolean('use_recurrence'),
        'target_update_interval': int(config['Algorithm'].getfloat('target_update_interval')),
        'min_steps_learn': int(config['Algorithm'].getfloat('min_steps_learn')),
        'eps_steps': int(config['Algorithm'].getfloat('eps_steps')),
        'double_dqn': config['Algorithm'].getboolean('double_dqn'),
        'learning_rate': config['Algorithm'].getfloat('learning_rate'),
        'clip_grad_norm': config['Algorithm'].getfloat('clip_grad_norm'),
        'delta_clip': None,  # default is squared error
        'OptimCls': torch.optim.Adam,
        'optim_kwargs': None,
        'initial_optim_state_dict': None,
    }

    # =====
    # Set up model
    img_channels = 1 if config['Env'].getboolean('grayscale') else 3
    model_kwargs = {
        'image_shape': (img_channels, img_len, img_len),
        'output_size': config['Env'].getint('num_arms'),
        'fc_size': config['Model'].getint('fc_size'),
        'lstm_size': config['Model'].getint('lstm_size'),
        'head_size': config['Model'].getint('head_size'),
        'dueling': config['Model'].getboolean('dueling'),
    }

    # =====
    # Set up logger
    # TODO set up the experiment name properly
    logger_kwargs = {
        'log_dir': log_dir,
        'name': config['Training']['exp_name'],
        'run_ID': config['Training'].getint('seed'),
        'snapshot_mode': config['Training']['log_snapshot_mode'],
    }

    # ==========
    # Initialize environment sampler
    sampler = SerialSampler(
        EnvCls=env_f,
        TrajInfoCls=TrajInfo,  # collect default trajectory info
        CollectorCls=CpuWaitResetCollector,  # each batch only has at most 1 episode
        env_kwargs=env_args,
        batch_T=config['Training'].getint('sampler_batch_T'),  # seq length of per batch of sampled data
        batch_B=1,
        max_decorrelation_steps=0,
        eval_env_kwargs=env_args,  # eval stuff, don't think it is used
        eval_n_envs=0,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    # ==========
    # Initialize algorithm, agent and model
    algo = R0D1(**algo_kwargs)
    agent = R2d1Agent(eps_init=config['Algorithm'].getfloat('eps_init'),
                      eps_final=config['Algorithm'].getfloat('eps_final'),
                      ModelCls=AtariR2d1Model,
                      model_kwargs=model_kwargs)

    # ==========
    # Initialize runner
    # (note can use MinibatchRlEval to also have eval runs)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(config['Training'].getfloat('total_num_steps')),
        log_interval_steps=int(config['Training'].getfloat('log_interval_steps')),
        affinity=affinity,
        seed=config['Training'].getint('seed'),
    )

    # ==========
    # Initialize logger and run experiment
    with logger_context(**logger_kwargs):
        runner.train()


if __name__ == "__main__":
    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config_path', type=str,
        default='/home/mila/c/chenant/repos/credit-transport/long_arms/r0d1/default_config.ini',
        help='path to the agent configuration .ini file'
    )
    # Logger parent path
    parser.add_argument(
        '--log_dir', type=str, default='./tmp_log',
        help='file path to the log file parent dir (default: ./tmp_log)'
    )

    # ======
    # Parse arguments
    args = parser.parse_args()
    print(args)

    # ======
    # Parse config
    config = configparser.ConfigParser()
    config.read(args.config_path)

    # =====================================================
    # Initialize GPU
    cuda_idx = 0 if torch.cuda.is_available() else None
    print('cuda_idx:', cuda_idx)

    # =====================================================
    # Set seeds
    seed = config['Training'].getint('seed')
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # =====================================================
    # Run experiment
    build_and_train(
        config=config,
        cuda_idx=cuda_idx,
        n_parallel=1,
        log_dir=args.log_dir,
    )
