#!/bin/bash
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=2:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=32000
#SBATCH --error=Error_Pangu_pytorch.log
#SBATCH --constraint=LSDF
#SBATCH --mail-user=unfuq@student.kit.edu

module purge
# module load jupyter/tensorflow eccodes-2.30.2_i22_ompi40

source ./venv/bin/activate

which python
echo "start"
python ./inference/test_main.py
# python ./finetune/lora_tune.py
echo "done"