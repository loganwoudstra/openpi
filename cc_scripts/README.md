To modify an existing training config to work for compute canada, check out pi05_franka in scr/openpi/training/config.py as a reference (need to modify paths, must manually download base models because CC nodes do not have access to the internet, and change fdsp devices to the number of GPUs you will request).

To schedule a training job, run the following command:
```
sbatch cc_scripts/train.sh [CONFIG] [EXPERIMENT_NAME] [RESUME_OR_OVERWRITE]

# example
sbatch cc_scripts/train.sh pi05_franka carrot_pick_and_place overwrite
```