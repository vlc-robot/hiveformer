from core.actioner import BaseActioner
from core.environments import RLBenchEnv
from typing import Tuple, Dict, List

import os
import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
import collections
import tap

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class Arguments(tap.Tap):
    microstep_data_dir: Path = "data/train_dataset/microsteps/seed0"
    keystep_data_dir: Path = "data/train_dataset/keysteps/seed0"

    tasks: Tuple[str, ...] = ("pick_up_cup",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")

    max_variations: int = 1
    offset: int = 0

    headless: bool = False
    gripper_pose: str = None
    max_tries: int = 10

    log_dir: Path = None


class KeystepActioner(BaseActioner):
    def __init__(self, keystep_data_dir) -> None:
        self.lmdb_env = lmdb.open(str(keystep_data_dir), readonly=True)
        self.lmdb_txn = self.lmdb_env.begin()

    def __exit__(self):
        self.lmdb_env.close()

    def reset(self, task_str, variation, instructions, demo_id):
        super().reset(task_str, variation, instructions, demo_id)

        value = self.lmdb_txn.get(demo_id.encode('ascii'))
        value = msgpack.unpackb(value)
        self.actions = value['action'][1:]

    def predict(self, taskvar_id, step_id, *args, **kwargs):
        out = {}
        if step_id < len(self.actions):
            out['action'] = self.actions[step_id]
        else:
            out['action'] = np.zeros((8, ), dtype=np.float32)
        print(self.demo_id, step_id, len(self.actions))
        return out


def evaluate_keysteps(args):
    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=args.cameras,
        headless=args.headless,
        gripper_pose=args.gripper_pose,
    )

    variations = range(args.offset, args.max_variations)

    taskvar_id = 0
    for task_str in args.tasks:
        for variation in variations:
            actioner = KeystepActioner(
                args.keystep_data_dir / f"{task_str}+{variation}")
            episodes_dir = args.microstep_data_dir / \
                task_str / f"variation{variation}" / "episodes"

            demo_keys, demos = [], []
            for ep in tqdm(episodes_dir.glob('episode*')):
                episode_id = int(ep.stem[7:])
                demo = env.get_demo(task_str, variation, episode_id)
                demo_keys.append(f'episode{episode_id}')
                demos.append(demo)

            success_rate = env.evaluate(
                taskvar_id,
                task_str,
                actioner=actioner,
                max_episodes=10,
                variation=variation,
                num_demos=len(demos),
                demos=demos,
                demo_keys=demo_keys,
                log_dir=args.log_dir,
                max_tries=args.max_tries,
                save_image=False,
            )

            print("Testing Success Rate {}: {:.04f}".format(
                task_str, success_rate))

if __name__ == '__main__':
    args = Arguments().parse_args()
    evaluate_keysteps(args)
