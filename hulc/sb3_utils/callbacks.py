from typing import Any, Dict

import gym
import torch as th
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            self._record_video(deterministic=True)
            self._record_video(deterministic=False)
        return True

    def _record_video(self, deterministic=True):
        screens = []
        behaviour_str = 'deterministic' if deterministic else 'stochastic'

        def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render(mode="rgb_array")
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            screens.append(screen.transpose(2, 0, 1))

        evaluate_policy(
            self.model,
            self._eval_env,
            callback=grab_screens,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=deterministic,
        )
        self.logger.record(
            f"trajectory/video_{behaviour_str}",
            Video(th.ByteTensor(np.array([screens])), fps=40),
            exclude=("stdout", "log", "json", "csv"),
        )
        
# model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
# video_recorder = VideoRecorderCallback(gym.make("CartPole-v1"), render_freq=5000)
# model.learn(total_timesteps=int(5e4), callback=video_recorder)


class SuccessRateCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, vlm_rewards=False, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.vlm_rewards = vlm_rewards
        self.log_freq = log_freq
        
        self.init_data()

    def init_data(self):
        self.data = {
            "is_success": [],
            "is_success_final": [],
            "reward": [],
            "tcp_z": [],
            "random": []
            }
        if self.vlm_rewards:
           self.data.update({
               "r_similarity": [],
               "r_prob": []
               })
        
    def _on_step(self) -> bool:
        assert len(self.locals['env'].envs) == len(self.locals["infos"])
        
        for i in range(len(self.locals['env'].envs)):
            # Log success
            self.data["is_success"] += [self.locals["infos"][i]["is_success"] * 1]
            # reward
            self.data["reward"] += [self.locals["infos"][i]["is_success"] * 1]
            # Log tcp_z
            self.data["tcp_z"] += [self.locals["infos"][i]['robot_info']['tcp_pos'][-1]]
            # Log random
            self.data["random"] += [np.random.rand()]
            # Log vlm reward
            if self.locals['dones'][i] == True:
                assert self.locals["n_steps"] > 1
                self.data["is_success_final"] += [self.locals["infos"][i]["is_success"] * 1]
                if self.vlm_rewards:
                    # r_similarity
                    self.data["r_similarity"] += [self.locals["infos"][i]['r_similarity']]
                    # r_prob
                    self.data["r_prob"] += [self.locals["infos"][i]['r_probs']]
            
        # Dump
        if self.num_timesteps % self.log_freq == 0:
            for key in self.data.keys():
                self.logger.record(f"custom_rollout/{key}", np.mean(self.data[key]))
            self.logger.dump(self.num_timesteps)
            self.init_data()
            
        return True