import os

def create_lsf_content(seed, epochs, lr, vae_epochs, vae_lr, gpu, hidden_dim, L, balance_lambda, lsd_threshold):
    content = f"""#BSUB -J KS
#BSUB -W 3:00
#BSUB -o /rsrch6/home/biostatistics/mpeng1/log/KS.out
#BSUB -e /rsrch6/home/biostatistics/mpeng1/log/KS.err
#BSUB -cwd /rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/true
#BSUB -q egpu-medium
#BSUB -gpu num=1:gmem=8 
#BSUB -u mpeng1@mdanderson.org
#BSUB -n 16
#BSUB -M 6
#BSUB -R rusage[mem=6]

module load cuda11.8/toolkit/11.8.0
module load python

python /rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/true/KS.py --seed {seed} --epochs {epochs} --lr {lr} --vae_epochs {vae_epochs} --vae_lr {vae_lr} --gpu {gpu} --hidden_dim {hidden_dim} --L {L} --balance_lambda {balance_lambda} --lsd_threshold {lsd_threshold}


"""
    return content

# Set the arguments for the Python script
output_dir = '/rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/true'
epochs = 2000
lr = 0.005
vae_epochs = 250
vae_lr = 0.01
gpu = 0
hidden_dim = 10
L = 2
balance_lambda = 1.0
seed = 1
lsd_threshold = 2

output_dir = '/rsrch6/home/biostatistics/mpeng1/PSLB/python/KS/true'

lsf_content = create_lsf_content(seed, epochs, lr, vae_epochs, vae_lr, gpu, hidden_dim, L, balance_lambda, lsd_threshold)
with open(os.path.join(output_dir, f'KS{seed}.lsf'), 'w') as lsf_file:
    lsf_file.write(lsf_content)
    



