from gym.vector.utils import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import torch.nn as nn
from snakeRLEnv import CustomEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th


def show_plot(state):
    plt.figure(figsize=(10, 8))
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:, :, idx], cmap='gray', extent=[0, 39, 39, 0])
        plt.xticks([0, 19, 20, 39])  # Marcamos valores en el eje x cada 1
        plt.yticks([0, 19, 20, 39])
        plt.grid(True, color='white', linewidth=0.5)
    plt.show()


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

# Crear el entorno y el modelo con la política personalizada
env = DummyVecEnv([lambda: CustomEnv(mode='human')])
env = VecFrameStack(env, 2, channels_order='last')
model = DQN('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
while True:
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()
        show_plot(state)
        # state, reward, done, info = env.step(env.action_space.sample())
