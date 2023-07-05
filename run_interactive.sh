
sinteractive --partition=deeplearn --time=24:00:00 --gres=gpu:v100:1 --ntasks=1 -A punim0478 --qos=gpgpudeeplearn --cpus-per-task=1 --mem=64G 

sinteractive --partition=deeplearn --time=24:00:00 --gres=gpu:v100sxm2:1 --ntasks=1 -A punim0478 --qos=gpgpudeeplearn --cpus-per-task=1 --mem=64G 


pip install jupyterlab
jupyter-notebook 

#Then, run the following command with port mapping on your laptop: 
ssh -N -L 6006:spartan-gpgpu088.hpc.unimelb.edu.au:8888 chunhua@spartan.hpc.unimelb.edu.au 

jupyter nbconvert static_embedding_baseline.ipynb --to script