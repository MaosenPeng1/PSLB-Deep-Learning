import os

# Function to generate LSF job submission script content
def create_lsf_content(queue_name, max_epochs, lr, vae_epochs, vae_lr, gpu, hidden_dim, L, balance_lambda, lsd_threshold):
    """
    Generates the content for the LSF job submission script.

    Args:
    queue_name (str): Name of the job queue (e.g., 'short', 'medium', etc.).
    max_epochs (int): Maximum number of training epochs for the main network.
    lr (float): Learning rate for the main network.
    vae_epochs (int): Number of training epochs for the VAE (Variational Autoencoder).
    vae_lr (float): Learning rate for the VAE.
    gpu (int): GPU device to use (0 or 1).
    hidden_dim (int): Size of the hidden layer in the network.
    L (int): Number of layers or latent factors.
    balance_lambda (float): Lambda value to balance the loss function.
    lsd_threshold (int): Threshold value for the LSD (latent space distance).

    Returns:
    str: Content for the LSF job submission script.
    """
    # Template for the LSF job script with placeholders for values
    content = f"""#BSUB -J lbc_net_job   # Job name
#BSUB -W 3:00          # Wall time (job duration limit)
#BSUB -o /path/to/log/lbc_net.out    # Output log file (change to your log directory)
#BSUB -e /path/to/log/lbc_net.err    # Error log file (change to your log directory)
#BSUB -cwd /path/to/working_directory   # Working directory (change to your directory)
#BSUB -q {queue_name}   # Queue name (specify the queue for job submission)
#BSUB -u your_email@example.com  # Email for job notifications
#BSUB -n 1   # Number of cores (set to 1)
#BSUB -M 6   # Memory allocation (6GB)
#BSUB -R rusage[mem=6]   # Memory usage request

module load python  # Load the Python module

# Run the Python script with specified parameters
python /path/to/working_directory/lbc_net.py \\
  --max_epochs {max_epochs} \\
  --lr {lr} \\
  --vae_epochs {vae_epochs} \\
  --vae_lr {vae_lr} \\
  --gpu {gpu} \\
  --hidden_dim {hidden_dim} \\
  --L {L} \\
  --balance_lambda {balance_lambda} \\
  --lsd_threshold {lsd_threshold}
    """
    return content

# Main script to generate the LSF file

# Define the script parameters (customize as needed)
output_dir = '/path/to/working_directory'  # Replace with your working directory
max_epochs = 20000   # The maximum number of training epochs for the main network
lr = 0.05        # Learning rate for the main network
vae_epochs = 250 # Number of training epochs for the VAE
vae_lr = 0.01    # Learning rate for the VAE
gpu = 0          # GPU device number to use (0 or 1)
hidden_dim = 100  # Size of hidden layers 
L = 2            # Number of latent layers
balance_lambda = 1.0   # Lambda value for loss balancing
lsd_threshold = 2    # LSD threshold for early stopping
queue_name = "e40short"   # Specify the queue for job submission (e.g., short, medium, long)

# Create the content for the LSF job submission script
lsf_content = create_lsf_content(queue_name, max_epochs, lr, vae_epochs, vae_lr, gpu, hidden_dim, L, balance_lambda, lsd_threshold)

# Write the LSF script to a file in the output directory
with open(os.path.join(output_dir, 'lbc_net_job.lsf'), 'w') as lsf_file:
    lsf_file.write(lsf_content)


