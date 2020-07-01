# ============================================================================
# "Long Arms" environment, toy example to evaluate forward information and
# backward value transport
#
# NOTE:
#   - Should include in the future the higher level dependencies as in
#     https://github.com/openai/gym/blob/master/docs/creating-environments.md
#
#
# Author: Anthony G. Chen
# ============================================================================


import gym
from gym import error, spaces, utils
from gym.utils import seeding

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10


class LongArmsEnv(gym.Env):
    """
    Description:
        The agent needs to pick between multiple arms, each with different
        rewards. The arm leads to a long hallway of partially observable
        timesteps before the reward is given.

        NOTE for now there are only two arms.

    Observation:

    Actions:

    States:
        Fully observable states describle by (arm #, arm steps), where (0,0)
        is the initial state (not in any arm). For example, (1,2) means the
        agent is twp states into the 1st arm. The agent receives image
        observations.

    """

    def __init__(self, dataset_path='./cifar_data/'):

        # State = (arm #, arm steps)
        self.corridor_length = None  # TODO set number
        self.num_arms = 2

        # ==
        self.action_space = spaces.Discrete(2)
        self.observation_space = None # TODO depends on image

        # ==
        # Download dataset


        tmp = self._init_observations(dataset_path)

        self.state = (0,0)

        pass

    def _init_observations(self, dataset_path):
        """
        Initialize image dataset corresponding to each observaiton



        Return: dict containing the initial, final and each arm's entrance
                image observation
                    {'init': PIL.Image.Image,
                     '1': PIL.Image.Image,
                     '2': PIL.Image.Image,
                     ...,
                     'final': PIL.Image.Image}

                sampler
        """

        # Define the image classes of each state
        init_class = 'automobile'
        arm_class = 'dog'
        corridor_class = 'bird'
        final_class = 'cat'

        # ==
        # Download / initialize dataset
        ds = CIFAR10(dataset_path, train=True, download=True)

        # ==
        # Separate the indeces for the different classes
        init_idxs, arm_idxs, corri_idxs, final_idxs = [], [], [], []
        (init_idx, arm_idx, corri_idx, final_idx) = (
            ds.class_to_idx[init_class], ds.class_to_idx[arm_class],
            ds.class_to_idx[corridor_class], ds.class_to_idx[final_class]
        )

        for i in range(len(ds)):
            current_class = ds[i][1]
            if current_class == init_idx:
                init_idxs.append(i)
            elif current_class == arm_idx:
                arm_idxs.append(i)
            elif current_class == corri_idx:
                corri_idxs.append(i)
            elif current_class == final_idx:
                final_idxs.append(i)
            else:
                pass

        # ==
        # Pick an image sample for the init, final and each arm states
        # Each sample has format: (PIL.Image.Image, class #)
        init_sample = ds[init_idxs[4]]
        final_sample = ds[final_idxs[64]]
        arm_samples = []
        for arm_j in range(self.num_arms):
            cur_arm_sample = ds[arm_idxs[2*arm_j]]
            arm_samples.append(cur_arm_sample)

        # Save samples as PIL images
        img_obs_dict = {
            'init': init_sample[0],
            'final': final_sample[0],
        }
        for arm_j in range(self.num_arms):
            img_obs_dict[str((arm_j+1))] = arm_samples[arm_j][0]

        # ==
        # Set up the image sampler for the corridor

        # TODO BELOW IS ALL WRONG CODE
        # Figure out when we have indeces and a original dataset (ds), how do
        # we get a simple way to sample data from it

        #corridor_ds = Subset(ds, corri_idxs)
        corridor_ds = ds.data[corri_idxs]
        data_loader = DataLoader(corridor_ds, batch_size=1, shuffle=True,
                                 num_workers=0)



        print(img_obs_dict)
        print(corridor_ds)
        #print(data_loader)

        tmp = next(corridor_ds)
        print(tmp)





    def step(self, action):

        # TODO assert action range?

        # ==
        # Update underlying state
        cur_arm, cur_step = self.state
        if cur_arm == 0:
            self.state = ((action + 1), 1)
        else:
            self.state = (cur_arm, cur_step + 1)

        # ==
        # Generate observation


        # ==
        # Generate reward, done and info

        # TODO


    def reset(self):
        self.state = ((action + 1), 1)

        # ==
        # Generate observation


        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # FOR TESTING ONLY
    print('hello')

    env = LongArmsEnv()
    print(env)
