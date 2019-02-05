PBS -lwalltime=00:30:00 ## time with hours:minutes:seconds
PBS -lselect=1:ncpus=1:mem=8gb:ngpus=1 # specify the gpu you need. the lower the memory requested, the higher the priority
module load anaconda3  # load anaconda3 or 2.
conda activate v-pytorch # activate your environment
python /rds/general/user/amp115/home/code/use_HPC/code_autoencoder/main.py #here is the full path to your code (mine as example)
