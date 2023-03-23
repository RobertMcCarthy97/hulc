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
import wandb
from wandb.integration.sb3 import WandbCallback


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]
    
def create_logger(args, env):
    # import pdb; pdb.set_trace()
    now = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    exp_path = f"./sb3_logs/exps/{args.exp_name}-{now}/"
    log_path = exp_path + f"{now}"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    
    callback_list = []
    callback_list += [EvalCallback(env, eval_freq=1000)]
    if args.track:
        callback_list += [
            WandbCallback(
                gradient_save_freq=10,
                model_save_path=None,
                verbose=1,
            ),]
    
    callback = CallbackList(callback_list)
    
    return new_logger, callback

def main():
    seed_everything(0, workers=True)  # type:ignore
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

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    # parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    
    parser.add_argument('-t', '--track', action='store_true', default=False)
    
    parser.add_argument("--exp-name", default='temp', type=str, help="Name of experiment.")

    args = parser.parse_args()

    assert "train_folder" in args

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

    env = None
    for checkpoint in checkpoints:
        epoch = get_epoch(checkpoint)
        model, env, _ = get_default_model_and_env(
            args.train_folder,
            args.dataset_path,
            checkpoint,
            env=env,
            device_id=args.device,
        )
        
        # wrap env
        env = CalvinTaskEnv(env, single_goal=True, visualize=args.debug)
        env = CalvinVLMEnv(env, model, use_vlm_reward=True, use_prob_reward=False, init_steps=10, use_model_actions=False)
        env = CalvinDictObsWrapper(env, ['robot_state'])
        
        # TODO: solve gym compatibility issues
        # TODO: don't use TimeLimit with VLM env!!!
        env = CustomTimeLimit(env, max_episode_steps=50)
        print("loaded env")
        
        if args.debug:
            # params
            use_model_actions = True
            
            # run env
            done = False
            i = 0
            obs = env.reset()
            print("reset env")
            while True:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                print(i)
                print(f"reward: {reward}, done: {done}, info.trunc: {info['TimeLimit.truncated']}")
                if done:
                    input()
                    i = 0
                    obs = env.reset()
                else:
                    i += 1
                    
        else:
            
            # W&B tracking
            if args.track:
                run = wandb.init(
                    project='CalvinVideoRewards',
                    group='Test',
                    entity='robertmccarthy11',
                    sync_tensorboard=True,
                    name=args.exp_name,
                    monitor_gym=False,  # auto-upload the videos of agents playing the game
                    save_code=False,  # optional
                )
            
            # Wrap the environment in a DummyVecEnv and normalize the observations
            env = DummyVecEnv([lambda: env])
            # env = VecVideoRecorder(env, f"videos/temp", record_video_trigger=lambda x: x % 2000 == 0, video_length=50, name_prefix=f"random-agent")
            env = VecMonitor(env, info_keywords=("success", "is_success"))
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            # logging
            logger, callback = create_logger(args, env)
            
            # Define and train the PPO agent
            model = PPO("MultiInputPolicy", env, verbose=1)
            model.set_logger(logger)
            model.learn(total_timesteps=int(1e5), callback=callback)


if __name__ == "__main__":
    main()