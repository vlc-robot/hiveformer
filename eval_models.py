from train_models import model_factory
from utils.misc import set_random_seed
from config.default import get_config
from core.actioner import BaseActioner
from core.environments import RLBenchEnv
from typing import Tuple, Dict, List

import os
import numpy as np
import itertools
from tqdm import tqdm
import copy
from pathlib import Path
import jsonlines
import tap

import torch
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torchvision.transforms.functional as transforms_f



class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'  # cpu, cuda

    eval_train_split: bool = False

    seed: int = 100  # seed for RLBench
    num_demos: int = 500

    headless: bool = False
    max_tries: int = 10
    save_image: bool = False
    record_video: bool = False


class Actioner(BaseActioner):
    def __init__(self, args) -> None:
        config = get_config(args.exp_config, args.extra_args)
        self.config = config

        self.device = torch.device(args.device)

        self.gripper_channel = self.config.MODEL.gripper_channel
        model_class = model_factory[config.MODEL.model_class]
        self.model = model_class(**config.MODEL)
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.use_history = config.MODEL.model_class == 'TransformerUNet'
        self.use_instr_embed = config.MODEL.use_instr_embed
        if type(config.DATASET.taskvars) is str:
            config.DATASET.taskvars = [config.DATASET.taskvars]
        self.taskvars = config.DATASET.taskvars

        if self.use_instr_embed != 'none':
            assert config.DATASET.instr_embed_file is not None
            self.lmdb_instr_env = lmdb.open(
                config.DATASET.instr_embed_file, readonly=True)
            self.lmdb_instr_txn = self.lmdb_instr_env.begin()
            self.memory = {'instr_embeds': {}}
        else:
            self.lmdb_instr_env = None

    def __exit__(self):
        self.lmdb_instr_env.close()

    def get_taskvar_instr_embeds(self, taskvar):
        instr_embeds = None
        if taskvar in self.memory['instr_embeds']:
            instr_embeds = self.memory['instr_embeds'][taskvar]

        if instr_embeds is None:
            instr_embeds = self.lmdb_instr_txn.get(taskvar.encode('ascii'))
            instr_embeds = msgpack.unpackb(instr_embeds)
            instr_embeds = [torch.from_numpy(x).float() for x in instr_embeds]
            # ridx = np.random.randint(len(instr_embeds))
            ridx = 0
            instr_embeds = instr_embeds[ridx]
            if self.use_instr_embed == 'avg':
                instr_embeds = torch.mean(instr_embeds, 0, keepdim=True)
            elif self.use_instr_embed == 'last':
                instr_embeds = instr_embeds[-1:]
            self.memory['instr_embeds'][taskvar] = instr_embeds
        return instr_embeds  # (num_ttokens, dim)

    def preprocess_obs(self, taskvar_id, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        rgb = torch.from_numpy(rgb).float().permute(0, 3, 1, 2)
        # # normalise to [-1, 1]
        # rgb = 2 * (rgb / 255.0 - 0.5)
        rgb = transforms_f.normalize(
            rgb.float(), 
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )

        if self.gripper_channel == "attn":
            gripper_imgs = torch.from_numpy(
                obs["gripper_imgs"]).float()  # (N, 1, H, W)
            rgb = torch.cat([rgb, gripper_imgs], dim=1)

        pcd = np.stack(obs['pc'], 0)  # (N, H, W, C)
        pcd = torch.from_numpy(pcd).float().permute(0, 3, 1, 2)

        batch = {
            'rgbs': rgb.unsqueeze(0),
            'pcds': pcd.unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'taskvar_ids': torch.LongTensor([taskvar_id]),
        }
        if self.use_instr_embed != 'none':
            taskvar = self.taskvars[taskvar_id]
            batch['instr_embeds'] = self.get_taskvar_instr_embeds(
                taskvar).unsqueeze(0)
            batch['txt_masks'] = torch.ones(
                1, batch['instr_embeds'].size(1)).long()

        if self.use_history:
            batch['rgbs'] = batch['rgbs'].unsqueeze(1)  # (B, T, N, C, H, W)
            batch['pcds'] = batch['pcds'].unsqueeze(1)
            batch['step_ids'] = batch['step_ids'].unsqueeze(1)
            batch['step_masks'] = torch.ones(1, 1)
            if len(self.history_obs) == 0:
                self.history_obs = batch
            else:
                for key in ['rgbs', 'pcds', 'step_ids', 'step_masks']:
                    self.history_obs[key] = torch.cat(
                        [self.history_obs[key], batch[key]], dim=1
                    )
            batch = copy.deepcopy(self.history_obs)

        # for k, v in batch.items():
        #     print(k, v.size())
        return batch

    def predict(self, taskvar_id, step_id, obs_state_dict):
        # print(obs_state_dict)
        batch = self.preprocess_obs(taskvar_id, step_id, obs_state_dict)
        with torch.no_grad():
            action = self.model(batch)[0]
        if self.use_history:
            action = action[-1]

        action = action.data.cpu().numpy()
        out = {
            'action': action
        }
        # print(self.demo_id, step_id)

        return out


def evaluate_keysteps(args):
    set_random_seed(args.seed)

    actioner = Actioner(args)

    config = actioner.config

    if args.eval_train_split:
        microstep_data_dir = Path(
            config.DATASET.data_dir.replace('keysteps', 'microsteps'))
        pred_dir = os.path.join(config.output_dir, 'preds', 'train')
    else:
        microstep_data_dir = ''
        pred_dir = os.path.join(config.output_dir, 'preds', f'seed{args.seed}')
    os.makedirs(pred_dir, exist_ok=True)

    env = RLBenchEnv(
        data_path=microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=config.DATASET.cameras,
        headless=args.headless,
        gripper_pose=config.MODEL.gripper_channel,
    )

    outfile = jsonlines.open(
        os.path.join(pred_dir, 'results.jsonl'), 'a', flush=True
    )
    for taskvar_id, taskvar in enumerate(actioner.taskvars):
        task_str, variation = taskvar.split('+')
        variation = int(variation)

        if args.eval_train_split:
            episodes_dir = microstep_data_dir / task_str / \
                f"variation{variation}" / "episodes"
            demo_keys, demos = [], []
            for ep in tqdm(episodes_dir.glob('episode*')):
                episode_id = int(ep.stem[7:])
                demo = env.get_demo(task_str, variation, episode_id)
                demo_keys.append(f'episode{episode_id}')
                demos.append(demo)
                # if len(demos) > 1:
                #     break
            num_demos = len(demos)
        else:
            demo_keys = None
            demos = None
            num_demos = args.num_demos

        success_rate = env.evaluate(
            taskvar_id,
            task_str,
            actioner=actioner,
            max_episodes=config.MODEL.max_steps,
            variation=variation,
            num_demos=num_demos,
            demos=demos,
            demo_keys=demo_keys,
            log_dir=Path(pred_dir),
            max_tries=args.max_tries,
            save_image=args.save_image,
            record_video=args.record_video,
        )

        print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
        outfile.write(
            {
                'checkpoint': config.checkpoint,
                'task': task_str, 'variation': variation,
                'num_demos': num_demos, 'sr': success_rate
            }
        )

    outfile.close()


if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    evaluate_keysteps(args)
