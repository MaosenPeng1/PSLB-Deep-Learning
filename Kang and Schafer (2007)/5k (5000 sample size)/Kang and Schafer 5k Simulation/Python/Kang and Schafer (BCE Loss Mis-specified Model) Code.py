import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PropensityDataset(Dataset):
    """
    A dataset class for storing dataset used to estimate propensity scores.

    Args:
        my_data (numpy.ndarray): The input data array containing treatment variable and covariates.

    Attributes:
        Z_norm (torch.Tensor): Normalized covariates.
        T (torch.Tensor): Treatment variable.
    """

    def __init__(self, my_data):
        Z = my_data[:, 6:10] 
        T = my_data[:, 1]

        Z = torch.from_numpy(Z).float() ## Convert the numpy arrays to PyTorch tensors
        self.Z_norm = (Z - Z.mean(dim=0)) / Z.std(dim=0) ## Normalization
        self.T = torch.from_numpy(T).long()

    def get_covariates(self):
        return self.Z_norm

    def __len__(self):
        return len(self.Z_norm)

    def __getitem__(self, idx):
        return self.T[idx], self.Z_norm[idx]
    
class LogisticNetwork(nn.Module):
    """
    A neural network model for propensity score estimation.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units in the network.
        output_dim (int, optional): The number of output units. Defaults to 1.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer after the first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        bn2 (nn.BatchNorm1d): Batch normalization layer after the second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        forward(x): Performs forward pass through the network.
        load_vae_encoder_weights(vae_encoder_weights): Loads weights from a VAE encoder.

    """

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LogisticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        residual = x
        x = torch.relu(self.fc2(x))
        x = x + residual
        x = self.sigmoid(self.fc3(x))
        return x
    
    def load_vae_encoder_weights(self, vae_encoder_weights):
        """
        Loads weights from a VAE encoder.

        Args:
            vae_encoder_weights (dict): The weights of the VAE encoder.

        """

        encoder_state_dict = vae_encoder_weights
        own_state = self.state_dict()
        encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if k in own_state} ## Filter out unnecessary keys
        own_state.update(encoder_state_dict)
        self.load_state_dict(own_state)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (nn.Sequential): Encoder network.
        decoder (nn.Sequential): Decoder network.

    Methods:
        reparameterize(mu, logvar): Reparameterizes the latent variables.
        forward(x): Performs a forward pass through the VAE.

    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()  
        self.encoder = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
          nn.Linear(latent_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, input_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent variables.

        Args:
            mu (torch.Tensor): Mean of the latent variables.
            logvar (torch.Tensor): Log variance of the latent variables.

        Returns:
            torch.Tensor: Reparameterized latent variables.

        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
      
    def forward(self, x):
        """
        Performs a forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed data.
            torch.Tensor: Mean of the latent variables.
            torch.Tensor: Log variance of the latent variables.

        """

        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """
    Calculates the loss function for a Variational Autoencoder (VAE).

    Parameters:
    - recon_x (torch.Tensor): The reconstructed input tensor.
    - x (torch.Tensor): The original input tensor.
    - mu (torch.Tensor): The mean of the latent space distribution.
    - logvar (torch.Tensor): The logarithm of the variance of the latent space distribution.

    Returns:
    - loss (torch.Tensor): The VAE loss, which is the sum of the reconstruction loss ((mean squared error)) and the KL divergence regularization term.
    """
    
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

    return reconstruction_loss + kl_divergence

def vae_train(vae_model, optimizer, dataloader, num_epochs):
    """
    Trains a Variational Autoencoder (VAE) model.

    Args:
        vae_model (nn.Module): The VAE model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        num_epochs (int): The number of training epochs.

    Returns:
        float: The average loss over the training epochs.
    """
    vae_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (treatment, inputs) in enumerate(dataloader):
            inputs, treatment = inputs.to(device), treatment.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = vae_model(inputs)
            loss = vae_loss(recon_batch, inputs, mu, logvar)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train(model, optimizer, criterion, dataloader, Z, device_input = "cpu"):
    """
    Trains the model using the given optimizer and dataloader.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion: The bce loss.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        Z (torch.Tensor): The input tensor for calculating ps.
        device_input (str, optional): The device to be used for training. Defaults to "cpu".

    Returns:
        tuple: A tuple containing the average loss, and ps.
    """
    
    model.train()

    total_loss = 0.0

    device = torch.device(device_input)

    for i, (treatment, inputs) in enumerate(dataloader):
        inputs, treatment = inputs.to(device), treatment.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, treatment.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    ps = model(Z).squeeze()
    
    avg_loss = total_loss / len(dataloader)
    avg_loss = torch.tensor([avg_loss])
    
    return avg_loss, ps

def load_data_with_seed(seed):
    """
    Load data from a file with a given seed.

    Parameters:
    seed (int): The seed used to generate the filename.

    Returns:
    numpy.ndarray or None: The loaded data as a numpy array, or None if the file is not found.
    """
    
    filename1 = f"KS5000_seed{seed}.RData.csv"

    try:
        my_data = np.genfromtxt(filename1, delimiter=',', skip_header=1)  
        return my_data

    except FileNotFoundError:
        print(f"Data file {filename1} not found.")
        return None


## Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load data
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True) 
args = parser.parse_args()
seed = args.seed
my_data = load_data_with_seed(seed)

## Set seed
seed_value = 100  ## Choose any seed value you prefer
torch.manual_seed(seed_value)

# Initialization parameters
batch_size = 32
input_dim = 4  ## Number of input features
hidden_dim = 10  ## Number of hidden units
latent_dim = 4 ## Dimensionality of the latent space
lr = 0.005 ## Learning rate
total_epochs = 20000

# Create PropensityDataset 
dataset = PropensityDataset(my_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)
vae_epoch = 250
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.01)
loss_ave = vae_train(vae_model, vae_optimizer, dataloader, vae_epoch)

## Train the propensity score model
model = LogisticNetwork(input_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.BCELoss()
# load the saved encoder weights
model.load_vae_encoder_weights(vae_model.encoder.state_dict()) ## Load the initial weights of the VAE encoder into the model

for epoch in range(total_epochs):

    loss_res, ps = train(model, optimizer, criterion, dataloader, dataset.get_covarites())

ps_list = []       
ps_list.append(ps)    
sys.stdout = sys.__stdout__

ps_np = np.array([ps.detach().numpy() for ps in ps_list]).T
df = pd.DataFrame(data=ps_np)
filename_ps = f"KS_ps_final_mis_bce{seed}.csv"
df.to_csv(filename_ps, index=False, header=False)


