#!/bin/sh
#SBATCH -o test1.out
#SBATCH --time=3000 #giới hạn thời gian chạy
#SBATCH --gres=gpu:0 # sử dụng node 0
#SBATCH -N 1 # số lượng node để chạy
#SBATCH --ntasks=1 # số lượng task để chạy
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

python model.py