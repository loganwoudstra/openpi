import shutil
import h5py
import numpy as np
import tyro
import glob
from pathlib import Path

import os
os.environ["HF_LEROBOT_HOME"] = "/mnt/data2/yi/logan/datasets/lerobot"

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

TASK = 'pick up the carrot and place it in the pot'

def main(data_dir: str):
    data_dir = Path(data_dir).resolve()
    REPO_NAME = data_dir.name
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
        
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 7D joint pos + gripper pos
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 6D ee rel vel + gripper pos
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    files = sorted(glob.glob(f"{data_dir}/*.hdf5"))
    for filepath in files:
        print(f"Processing {filepath}")
        with h5py.File(filepath, "r") as f:
            qpos = f["/observations/qpos"][:]        # (T, 7)
            gripper_pos = f["/observations/gpos"][:] # (T, 1)
            state = np.concatenate([qpos, gripper_pos], axis=-1)  # (T, 8)
            actions = f["/action"][:]                 # (T, 7)
            images = f["/observations/images/ext1"][:]  # (T, 480, 640, 3)
            wrist = f["/observations/images/wrist"][:]  # (T, 480, 640, 3)
            task = TASK

            T = qpos.shape[0]
            for t in range(T):
                dataset.add_frame(
                    {
                        "image": images[t],
                        "wrist_image": wrist[t],
                        "state": state[t].astype(np.float32),
                        "actions": actions[t].astype(np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)
