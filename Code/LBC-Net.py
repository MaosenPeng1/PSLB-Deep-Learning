import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse

from functions import *
import time

parser = argparse.ArgumentParser(description='LBC-Net Estimation')

## for dataset
parser.add_argument('--seed', type=int, default=1, help='seed number')

## for model
parser.add_argument('--gpu', type=int, default=0, help='gpu ids(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--hidden_dim', type=int, default=10, help='hidden dimension (default: 10)')
parser.add_argument('--L', type=int, default=2, help='number of hidden layers - 1 (default: 2)')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
parser.add_argument('--vae_epochs', type=int, default=250, help='number of epochs to train VAE (default: 250)')
parser.add_argument('--vae_lr', type=float, default=0.01, help='learning rate for VAE (default: 0.01)')
parser.add_argument('--balance_lambda', type=float, default=1.0, help='balance parameter (default: 1.0)')
parser.add_argument('--lsd_threshold', type=float, default=2, help='threshold for LSD (default: max 2%), stopping criterion')

args = parser.parse_args()
print(args)

## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

def main():
    start_time = time.time()
    ## load data
    filename = f"data_seed{args.seed}.csv"
    data = pd.read_csv(filename, delimiter=',')

    ## Load ck and h
    ck_h_file = pd.read_csv(f"ck_h{args.seed}.csv")
    ck = ck_h_file['ck'].values
    h = ck_h_file['h'].values
    ck = torch.tensor(ck, dtype=torch.float32).to(device)
    h = torch.tensor(h, dtype=torch.float32).to(device)

    ## set seed
    torch.manual_seed(100)  ## Choose any number you prefer, here we use 100
    if torch.cuda.is_available():
        torch.cuda.manual_seed(100) 
        torch.cuda.manual_seed_all(100)
        torch.backends.cudnn.deterministic = True

    Z = data.iloc[:, 2:6].values ## Specify covariates columns
    T = data.iloc[:, 1].values ## Specify Treatment column
    n, p = Z.shape
    
    Z_norm = (Z - Z.mean(axis=0)) / Z.std(axis=0) ## Normalization
    Z_norm = np.concatenate((np.ones((Z_norm.shape[0], 1)), Z_norm), axis=1) ## Add intercept for standardization in the balance loss function
    
    Z_norm = torch.tensor(Z_norm, dtype=torch.float32).to(device)
    T = torch.tensor(T, dtype=torch.float32).to(device)
    p = p+1
    
    # Train the VAE model
    vae_model = VAE(p, p).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.vae_lr)
    vae_model.train()

    for epoch in range(args.vae_epochs):
        
        vae_optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(Z_norm)
        loss = vae_loss(recon_batch, Z_norm, mu, logvar)
        loss.backward()
        vae_optimizer.step()
    
    ps_model = PropensityNetwork(p, args.hidden_dim, args.L).to(device)
    optimizer = optim.Adam(ps_model.parameters(), lr=args.lr, weight_decay=1e-5)
    ps_model.load_vae_encoder_weights(vae_model.encoder.state_dict()) ## Load the initial weights of the VAE encoder into the model

    for epoch in range(args.epochs):
        ps_model.train()

        optimizer.zero_grad()

        outputs = ps_model(Z_norm).squeeze()
                  
        penalty = penalty_loss(outputs, T, ck, h)
        balance_loss, LSD = local_balance_ipw_loss(outputs, T, Z_norm, ck, h, device)
        loss = args.balance_lambda * penalty + balance_loss
        loss.backward()
        optimizer.step()

        if LSD < args.lsd_threshold:
            break
    print(f"Current epoch: {epoch}, MAX LSD: {LSD}%")
    print(f"Epoch {epoch}: Loss: {loss}, Penalty: {penalty}, Balance Loss: {balance_loss}")
    ps = ps_model(Z_norm).squeeze().detach().cpu().numpy()
    df = pd.DataFrame(data=ps)
    df.to_csv(f"ps{args.seed}.csv", index=False, header=False)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time is: {execution_time} seconds")

if __name__ == "__main__":
    main()


