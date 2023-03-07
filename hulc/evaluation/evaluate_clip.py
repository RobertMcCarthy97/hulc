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
from torch.utils.data import Subset
from torch.utils.data import DataLoader


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
    return unique_lang_idxs

def get_specific_batch(dataset, indices):
    idxs_new = [int(i) for i in indices]
    my_subset = Subset(dataset, idxs_new)
    # import pdb; pdb.set_trace()
    loader = DataLoader(my_subset, batch_size=len(idxs_new), shuffle=False) 
    batch = next(iter(loader))
    return batch

def visualize_batch(dataset, batch):
    import matplotlib.pyplot as plt
    import cv2
    def play_sequence(seq, text):
        n_frames = seq.shape[0]
        plt.figure()
        for i in range(n_frames):
            frame = seq[i]
            frame = 255 * (frame + 1) / 2
            frame = np.moveaxis(frame, 0, -1)
            frame = cv2.resize(frame, (500, 500))
            frame = frame[:, :, ::-1]
            cv2.imshow(text, frame)
            cv2.waitKey(1)
            # plt.imshow(frame)
            # import pdb; pdb.set_trace()
            input()
        cv2.destroyAllWindows()
    
    size = batch['robot_obs'].shape[0]
    assert size == 7
    for i in range(size):
        text = dataset.lang_ann_str[i]
        rgb_static_sequence = batch['rgb_obs']['rgb_static'][i]
        print(f"ANNOTATION: \n{text}\n")
        input("[ENTER] to play sequence")
        play_sequence(rgb_static_sequence.numpy(force=True), text)
        
    

def embed_batch_list(model, batch):
    def convert_dict_to_cuda(torch_dict):
        cuda_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, dict):
                cuda_dict[key] = convert_dict_to_cuda(value)
            else:
                cuda_dict[key] = value.cuda()
        return cuda_dict
    
    batch = convert_dict_to_cuda(batch)
    
    # perceptual emb
    perceptual_emb = model.perceptual_encoder(
        batch["rgb_obs"], batch["depth_obs"], batch["robot_obs"]
    )
    # visual features
    pr_state, seq_vis_feat = model.plan_recognition(perceptual_emb)
    # lang features
    encoded_lang = model.language_goal(batch["lang"])
    import pdb; pdb.set_trace()
    
    # image, lang features
    image_features, lang_features = model.proj_vis_lang(seq_vis_feat, encoded_lang)
    
    #### CLIP loss
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    # # symmetric loss function
    # labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
    # loss_i = cross_entropy(logits_per_image, labels)
    # loss_t = cross_entropy(logits_per_text, labels)
    # loss = (loss_i + loss_t) / 2
    
    import pdb; pdb.set_trace()
        
    

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
        # import pdb; pdb.set_trace()
        batch_idxs = get_lang_idxs(data_module.val_datasets['lang'])
        batch = get_specific_batch(data_module.val_datasets['lang'], batch_idxs)
        # visualize_batch(data_module.val_datasets['lang'], batch)
        embed_batch_list(model, batch)
        # evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)
        
        
if __name__ == "__main__":
    main()