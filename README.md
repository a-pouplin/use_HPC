## Access to the High Performance Computing 
Each student has been assigned to the [HPC services](http://www.imperial.ac.uk/computational-methods/cm-hub/hpc-guide/).  Make sure you are connected to the Imperial network if you are working from home, using a [VPN](https://www.imperial.ac.uk/admin-services/ict/self-service/connect-communicate/remote-access/method/set-up-vpn/).

* connect to your container: `ssh <user_name>@login.cx1.hpc.ic.ac.uk`
* you should now be in: `rds/general/user/<user_name>/home`

## Download python with anaconda
The containers provided by HPC already have all the modules you need to do your project. If you need a specific software, first check if it is available using `module avail`, and discuss it with the HPC services directly. 

* If you prefer to use python 3: `module load anaconda3/4.3.1`
* `anaconda-setup`
* Launch Python: `python`

## Create your virtual environment
A virtual environnment is like a small container where you can install all the packages you need for a project. You can create as many virtual environments as you like [with conda](https://conda.io/docs/user-guide/tasks/manage-environments.html)
* `conda create -n <name environment> python=x.x` 
* `source activate <name environment>`: to activate your environment.
* `source deactivate`: to deactivate your environment.

Install your packages (ex: pytorch): 
* `conda activate <name_environment>`
* `conda install pytorch torchvision -c pytorch`
* `conda install matplotlib` ...

## Recover and update your code 
You can either use git or bitbucket.

For example, if you don't have (yet) any code:
* `git clone https://github.com/a-pouplin/use_HPC.git`

## Submit your code to the system
In order to submit your job, you will need to submit a [PBS file](https://en.wikipedia.org/wiki/Portable_Batch_System) using a [bashscript](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) to HPC. Your job will be in a queue (which can last either a few minutes or a few hours). You can create your bashscript using vim: `vim <pbs_name>.sh`

You will need to write some specifications:
```
#PBS -lwalltime=00:30:00
#PBS -lselect=1:ncpus=1:mem=8gb:ngpus=1
module load anaconda3/4.3.1
source activate <name_environment>
python $HOME/<path_to_code>/<code>.py
```
* `qsub <pbs_name>`: submit your job in the queue
*

## Check the queue and your job status
* `qstat -s`: check your job status
* `qstat -q`: check the queue information (which GPU/ CPU are available)
* `qdel <job_number>`: delete the current job

You will end up with two output: 
* `<pbs_name>.sh.o<job_number>`: output file, containing all your print in your python job
* `<pbs_name>.sh.e<job_number>`: error file, containing an error, if your job failed


## Transfer your data from your computer to HPC 
Dowload your file with [scp](https://en.wikipedia.org/wiki/Secure_copy) (you can also use FileZilla): 
* `scp -r <user_name>@login.cx1.hpc.ic.ac.uk/<path_to_your_experiments> <path_on_your_computer>`

## If you need any help
You can ask us, or go directly to the weekly HPC drop-in clinic: every Tuesday from 14:00-16:00 in room 402, 4th Floor Sherfield Building. You can also have a look at their [Getting started guide](http://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/getting-started/)

