import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from snakeRLEnv import CustomEnv
from gym.vector.utils import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

CHECKPOINT_DIR = './train'
LOG_DIR = './logs'


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 2  # Get the number of input channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path_base = save_path
        self.run_counter = self._get_last_run_counter() + 1

    def _get_last_run_counter(self):
        existing_runs = [d for d in os.listdir(self.save_path_base) if d.startswith("run")]
        run_numbers = [int(run.replace("run", "")) for run in existing_runs]
        return max(run_numbers, default=0)

    def _init_callback(self):
        self.save_path = os.path.join(self.save_path_base, f'run{self.run_counter}')
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)


if __name__ == '__main__':
    env = DummyVecEnv([lambda: CustomEnv()])  # Wrap your custom environment with DummyVecEnv
    env = VecFrameStack(env, 2, channels_order='last')

    hyperparams = {'buffer_size': 100000,
                   'learning_rate': 0.0001,
                   'batch_size': 32,
                   'learning_starts': 100000,
                   'target_update_interval': 1000,
                   'train_freq': 4,
                   'gradient_steps': 1,
                   'exploration_fraction': 0.1,
                   'exploration_final_eps': 0.01,
                   # If True, you need to deactivate handle_timeout_termination
                   # in the replay_buffer_kwargs
                   'optimize_memory_usage': False}

    callback = TrainAndLoggingCallback(check_freq=200000, save_path=CHECKPOINT_DIR)
    model = DQN('CnnPolicy', env, **hyperparams, verbose=1, tensorboard_log=LOG_DIR, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=30000000, callback=callback)
