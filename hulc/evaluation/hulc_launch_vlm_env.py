import numpy as np
import gym

import argparse
import logging
from pathlib import Path
import sys
from os import system

# This is for using the locally installed repo clone when using slurm
# from calvin_agent.evaluation.evaluate_policy_llm import evaluate_policy

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.utils import get_default_model_and_env
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
from pytorch_lightning import seed_everything

from calvin_agent.evaluation.calvin_vlm_env import CalvinTaskEnv, CalvinVLMEnv, CalvinDictObsWrapper, CustomTimeLimit

# logger = logging.getLogger(__name__)

from datetime import datetime

# sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, VecVideoRecorder
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps, EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from hulc.sb3_utils.callbacks import VideoRecorderCallback, SuccessRateCallback


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )
    # parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")
    # Relevant args
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--debug", action="store_true", default=False, help="Print debug info and visualize environment.")
    parser.add_argument('-t', '--track', action='store_true', default=False)
    parser.add_argument("--exp-name", default='temp', type=str, help="Name of experiment.")
    parser.add_argument("--exp-group", default='temp', type=str, help="Name of experiment group.")
    parser.add_argument("--vlm-rewards", action='store_true', default=False, help="Whether to use VLM rewards.")
    
    args = parser.parse_args()
    assert "train_folder" in args
    
    return args

def load_calvin_items(args):
    checkpoints = []
    if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
        print("Evaluating model with last checkpoint.")
        checkpoints = [get_last_checkpoint(Path(args.train_folder))]
    elif args.checkpoints is not None:
        print(f"Evaluating model with checkpoints {args.checkpoints}.")
        checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
    elif args.checkpoints is None and args.last_k_checkpoints is not None:
        print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
        checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
    elif args.checkpoint is not None:
        checkpoints = [Path(args.checkpoint)]
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    
    epoch = get_epoch(checkpoint)
    model, env, _ = get_default_model_and_env(
        args.train_folder,
        args.dataset_path,
        checkpoint,
        env=None,
        device_id=args.device,
    )
    
    return model, env, None

def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]
    
def test_rollout(args, env):
    model = PPO.load("/home/robert/Research/rl_envs/hulc/sb3_logs/exps/temp-24_03_2023-15_47_22/best_model", env=env)
    # run env
    done, i = False, 0
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"[{i}] reward: {reward}, done: {done}") #", info.trunc: {info['TimeLimit.truncated']}")
        if done:
            input()
            i = 0
            obs = env.reset()
        else:
            i += 1

def make_vlm_env(args, env, model):
    # wrap env
    env = CalvinTaskEnv(env, single_goal=True, visualize=args.debug)
    if args.vlm_rewards:
        env = CalvinVLMEnv(env, model, use_vlm_reward=True, use_prob_reward=False, init_steps=10, use_model_actions=False)
    else:
        # Don't use TimeLimit with VLM env
        env = CustomTimeLimit(env, max_episode_steps=50)
    env = CalvinDictObsWrapper(env, ['robot_state'])
    # TODO: solve gym compatibility issues
    return env
    
def create_logger(args, env):
    # TODO: use a better naming scheme!!
    now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_path = f"./sb3_logs/exps/{args.exp_name}-{now}/"
    log_path = exp_path + f"{now}"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    
    callback_list = []
    callback_list += [EvalCallback(env, eval_freq=1000, best_model_save_path=exp_path)]
    callback_list += [SuccessRateCallback(vlm_rewards=args.vlm_rewards, log_freq=1000)]
    callback_list += [VideoRecorderCallback(env, render_freq=10000, n_eval_episodes=1, deterministic=True)]
    if args.track:
        callback_list += [
            WandbCallback(
                gradient_save_freq=10,
                model_save_path="saved_models/",
                model_save_freq=1000,
                verbose=1,
            ),]
    callback = CallbackList(callback_list)
    
    return new_logger, callback, exp_path

def main():
    seed_everything(1, workers=True)  # type:ignore
    
    args = get_args()
    model, env, _ = load_calvin_items(args)
    # custom env
    env = make_vlm_env(args, env, model)
        
    if args.debug:
        test_rollout(args, env)
                
    else:
        # W&B tracking
        # TODO: implement sweeps??
        if args.track:
            run = wandb.init(
                entity='robertmccarthy11',
                project='CalvinVideoRewards',
                group=args.exp_group,
                name=args.exp_name,
                sync_tensorboard=True,
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=False,  # optional
            )
            wandb.log({'args': vars(args)})
        
        # Wrap the environment in a DummyVecEnv and normalize the observations
        env = DummyVecEnv([lambda: env])
        env = VecMonitor(env, info_keywords=("success", "is_success",))
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        # logging
        logger, callback, exp_path = create_logger(args, env)
        
        # Define and train the PPO agent
        model = PPO("MultiInputPolicy", env, verbose=1)
        model.set_logger(logger)
        model.learn(total_timesteps=int(5e5), callback=callback)


if __name__ == "__main__":
    main()