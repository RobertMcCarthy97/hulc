import argparse
import logging
from pathlib import Path
import sys

# This is for using the locally installed repo clone when using slurm
from calvin_agent.evaluation.evaluate_policy import evaluate_policy

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.utils import get_default_model_and_env
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
from pytorch_lightning import seed_everything

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def get_lang_idxs(dataset):

    # self.lang_ann_str
    # self.episode_lookup, self.lang_lookup, self.lang_ann
    unique_lang_idxs = []
    for i in range(np.array(dataset.lang_lookup).max()):
        lang_i = np.where(np.equal(dataset.lang_lookup, i))[0][0]
        unique_lang_idxs += [lang_i]
    unique_lang_idxs = np.array(unique_lang_idxs)
    return unique_lang_idxs

def get_batch_defined_idxs(dataset, idxs):
    assert len(idxs.shape) == 1
    batch_list = []
    for i in idxs:
        batch_list += [dataset[int(i)]]
    
    return batch_list

def embed_batch_list(batch_list):
    for sample in batch_list:
        
        
    

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

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
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
        model, env, data_module = get_default_model_and_env(
            args.train_folder,
            args.dataset_path,
            checkpoint,
            env=env,
            device_id=args.device,
        )
        batch_idxs = get_lang_idxs(data_module.val_datasets['lang'])
        get_batch_defined_idxs(data_module.val_datasets['lang'], batch_idxs)
        import pdb; pdb.set_trace()
        evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)
        
        
if __name__ == "__main__":
    main()