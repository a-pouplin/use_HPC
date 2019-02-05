#PBS -lwalltime=00:30:00
#PBS -lselect=1:ncpus=1:mem=8gb:ngpus=1
module load anaconda3
conda activate v-pytorch
python /rds/general/user/amp115/home/code/use_HPC/code_autoencoder/main.py 
