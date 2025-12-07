#!/bin/bash 
#SBATCH -A IscrC_ConFT
#SBATCH -p boost_usr_prod
#SBATCH --time 22:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1      # 4 gpus per node out of 4
#SBATCH --mem=200GB         # memory per node out of 494000MB 
#SBATCH --job-name=inflora
#SBATCH --output=/leonardo_scratch/fast/IscrC_PECFT/luigi/InfLoRA/outputs/exp_%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_PECFT/luigi/InfLoRA/outputs/exp_%j.err
module load python/3.9.5
module load cuda/12.1
module load gcc/11.3.0
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
export CUDA_LAUNCH_BLOCKING=1
source /leonardo_scratch/fast/IscrC_PECFT/luigi/pyenvs/inflora_env/bin/activate

scenario=$1
n_tasks=$2
rank=$3
epochs=$4
seeds=(99292 1235 6132)
#seeds=(42 45 46)
json_seeds=$(printf '%s\n' "${seeds[@]}" | jq -s '.')

og_config_file="configs/luigi_configs/${scenario}_${n_tasks}t_inflora.json"
spec_config_file="outputs/${scenario}/${n_tasks}t_${rank}r_${epochs}e.config"
output_file="outputs/${scenario}/${n_tasks}t_${rank}r_${epochs}e_${seeds[1]}s.out"
tmp_file=$( date '+%F_%H:%M:%S' )

# Update the seed and num_epochs in the JSON file
jq --argjson seeds "$json_seeds" --argjson num_epochs $epochs  --argjson rank $rank '.seed = $seeds | .init_epoch = $num_epochs | .epochs = $num_epochs | .rank = $rank' "$og_config_file" > $spec_config_file

echo $json_seeds

python main.py --config $spec_config_file > $output_file 2>&1
