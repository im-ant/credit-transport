# =============================================================================
# Training the agent in Long Arms environment
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import configparser
import math
import random  # only imported to set seed
import sys

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

sys.path.insert(0, './ElectrophysRL/')  # I'm sorry
from dopatorch.agents.a2c.a2c_agent import A2CAgent
from envs.long_arms import LongArmsEnv


# TODO VERY IMPORTANT TODO TODO
# FIGURE OUT A BETTER PRACTICE FOR IMPORTING THE DOPATORCH MODULE


# ========================================
# Helper methods for agent initialization
# ========================================


def init_env(config: configparser.ConfigParser):
    img_len = config['Env'].getint('img_len')

    env = LongArmsEnv(
        corridor_length=config['Env'].getint('corridor_length'),
        img_size=(img_len, img_len),
        grayscale=config['Env'].getboolean('grayscale'),
        flatten_obs=config['Env'].getboolean('flatten_obs'),
        scale_observation=config['Env'].getboolean('scale_observation'),
        dataset_path=config['Env']['dataset_path']
    )

    return env


def init_agent(config: configparser.ConfigParser, env, device='cpu'):
    """
    Method for initializing the agent
    :return: agent instance
    """

    # ===
    # Initialize agent
    agent_type = config['Agent']['type']

    if agent_type == 'a2c':
        agent = A2CAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            observation_dtype=torch.float,
            gamma=config['Agent'].getfloat('gamma'),
            use_recurrent_net=config['Agent'].getboolean('use_recurrent_net'),
            num_rollout_steps=config['Agent'].getint('num_rollout_steps'),
            value_loss_coef=config['Agent'].getfloat('value_loss_coef'),
            entropy_coef=config['Agent'].getfloat('entropy_coef'),
            max_grad_norm=config['Agent'].getfloat('max_grad_norm'),
            use_acktr=config['Agent'].getboolean('use_acktr'),
            device=device
        )
    else:
        raise NotImplementedError

    return agent


# ========================================
# Run environment
# ========================================

def run_environment(config: configparser.ConfigParser,
                    device: str = 'cpu',
                    logger: torch.utils.tensorboard.SummaryWriter = None):
    # =========
    # Set up environment
    config_seed = config['Training'].getint('seed')
    env = init_env(config)
    env.seed(config_seed)

    # =========
    # Set up agent
    agent = init_agent(config, env, device=device)

    # =========
    # Start training

    # Extract training variables
    config_num_episodes = config['Training'].getint('num_episode')
    config_record_video = config['Video'].getboolean('record')
    config_video_freq = config['Video'].getint('frequency')
    config_video_maxlen = config['Video'].getint('max_length')
    config_video_fps = config['Video'].getint('fps')

    # Train
    print(f'Starting training, {config_num_episodes} episodes')
    for episode_idx in range(config_num_episodes):
        # ==
        # Reset environment and agent
        observation = env.reset()
        action = agent.begin_episode(observation)

        # Counters
        cumu_reward = 0.0
        timestep = 0

        # ==
        # (optional) Record video
        video = None
        if config_record_video:
            if episode_idx % int(config_video_freq) == 0:
                # Render first frame and insert to video array
                frame = env.render()
                video = np.zeros(shape=((config_video_maxlen,) + frame.shape),
                                 dtype=np.uint8)  # (max_vid_len, C, W, H)
                video[0] = frame

        # ==
        # Run episode
        while True:
            # ==
            # Interact with environment
            observation, reward, done, info = env.step(action)
            action = agent.step(observation, reward, done)

            # ==
            # Counters
            cumu_reward += reward
            timestep += 1

            # ==
            # Optional video
            if video is not None:
                if timestep < config_video_maxlen:
                    video[timestep] = env.render()

            # ==
            # Episode done
            if done:
                # Logging
                if logger is not None:
                    # Add reward
                    logger.add_scalar('Reward', cumu_reward,
                                      global_step=episode_idx)
                    # Optionally add video
                    if video is not None:
                        # Determine last frame
                        last_frame_idx = timestep + 2
                        if last_frame_idx > config_video_maxlen:
                            last_frame_idx = config_video_maxlen

                        # Change to tensor
                        vid_tensor = torch.tensor(video[:last_frame_idx, :, :, :],
                                                  dtype=torch.uint8)
                        vid_tensor = vid_tensor.unsqueeze(0)

                        # Add to tensorboard
                        logger.add_video('Run_Video', vid_tensor,
                                         global_step=episode_idx,
                                         fps=config_video_fps)

                    # Occasional print
                    if episode_idx % 100 == 0:
                        print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                else:
                    print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                # Agent logging TODO: not sure if this is the best practice
                agent.report(logger=logger, episode_idx=episode_idx)
                break

            # TODO: have some debugging print-out (e.g. every 100 episode) to make sure times and
            # things are good and training is happening

    env.close()
    if logger is not None:
        logger.close()


if __name__ == "__main__":

    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='RL in discrete environment')

    # ==
    # Environment, agent and other training parameters
    parser.add_argument(
        '--config_path', type=str,
        default='/home/mila/c/chenant/repos/credit-transport/long_arms/default_config.ini',
        help='path to the agent configuration .ini file'
    )

    # ==
    # I/O parameters
    parser.add_argument('--log_dir', type=str, default=None,
                        help='file path to the log file (default: None, printout instead)')
    parser.add_argument('--tmpdir', type=str, default='./',
                        help='temporary directory to store dataset for training (default: cwd)')

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # =====================================================
    # Set seeds
    seed = config['Training'].getint('seed')
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # =====================================================
    # Initialize logging
    if args.log_dir is not None:
        # Tensorboard logger
        logger = SummaryWriter(log_dir=args.log_dir)

        """
        TODO FIX THIS
        Need to add agent parameters along with the env parameters, not sure
        how to incorporate this in a nice way into tensorboard yet
        
        # Add hyperparameters
        logger.add_hparams(hparam_dict=vars(args), metric_dict={})
        """
    else:
        logger = None

    # =====================================================
    # Start environmental interactions
    run_environment(config, device=device, logger=logger)
