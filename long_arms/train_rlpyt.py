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

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
# from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from rlpyt.models.dqn.atari_r2d1_model import AtariR2d1Model
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from envs.long_arms import LongArmsEnv
from r0d1.algo_r0d1 import R0D1


def env_f(**kwargs):
    return GymEnvWrapper(LongArmsEnv(**kwargs))


def build_and_train(run_ID=0, cuda_idx=None, n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")

    # ==
    # Env
    env_args = {'corridor_length': 5,
                'img_size': (20, 20),
                'grayscale': True,
                'flatten_obs': False,
                'scale_observation': False,
                'dataset_path': '/network/tmp1/chenant/ant/dataset/cifar/'}

    # ==
    # Sampler
    sampler = SerialSampler(
        EnvCls=env_f,
        TrajInfoCls=TrajInfo,  # default traj info + GameScore
        CollectorCls=CpuWaitResetCollector,  # don't immediate reset episode
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=10,  # seq length of per batch of sampled data
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=0,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    # ==
    # Algorithm and agent

    # Configure in:
    # rlpyt/models/dqn/atari_catdqn_model.py
    model_kwargs = {'image_shape': (1, 20, 20),
                    'output_size': 2}
    algo_kwargs = {'batch_T': 3, 'batch_B': 2, 'warmup_T': 1,
                   'store_rnn_state_interval': 4,
                   'min_steps_learn': 16,
                   'replay_size': 10, 'replay_ratio': 1,
                   'target_update_interval': 1, 'n_step_return': 2,
                   'learning_rate': 0.0001, 'eps_steps': 10000,
                   'double_dqn': False, 'prioritized_replay': False,
                   'input_priorities': False, 'updates_per_sync': 1}
    algo = R0D1(**algo_kwargs)
    agent = R2d1Agent(ModelCls=AtariR2d1Model,
                      model_kwargs=model_kwargs)


    # ==
    # Runner
    # can also use MinibatchRlEval
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=(50e6),  # total steps to train I think
        log_interval_steps=1000,
        affinity=affinity,
    )
    config = dict(game='LongArms')
    name = "a2c_" + 'LongArms'
    log_dir = "example_3"

    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=1)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
    )
