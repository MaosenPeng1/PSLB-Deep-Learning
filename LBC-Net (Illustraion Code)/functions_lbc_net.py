import torch
import torch.nn as nn

class lbc_net(nn.Module):
    def __init__(self, input_dim, hidden_dim = 10, L = 2): ## L is the number of hidden layers - 1
        super(lbc_net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L

        self.initial_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_bn = nn.BatchNorm1d(self.hidden_dim)
        self.initial_activation = nn.ReLU()
    
        # Define the middle layers
        self.middle_layers = nn.Sequential(
            *(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU()
            ) for _ in range(self.L - 1))
        )

        # Final layer
        self.final_layer = nn.Linear(self.hidden_dim, 1)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x, epsilons = 0.001):
        # Initial layer forward pass
        x = self.initial_layer(x)
        x = self.initial_bn(x)
        x = self.initial_activation(x)
        
        # Middle layers forward pass
        for layer in self.middle_layers:
            x = layer(x)

        # Final layer forward pass
        x = self.final_layer(x)
        x = epsilons + (1 - 2*epsilons) * self.output_activation(x) # modified sigmoid function, guarantee overlap assumption

        return x

    def load_vae_encoder_weights(self, vae_encoder_weights):
        """
        Load weights from a pre-trained VAE encoder into the network.

        Args:
        vae_encoder_weights (dict): Dictionary containing the VAE encoder weights.
        """
        own_state = self.state_dict()
        encoder_state_dict = {k: v for k, v in vae_encoder_weights.items() if k in own_state}
        own_state.update(encoder_state_dict)
        self.load_state_dict(own_state)

class vae(nn.Module):
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

    def __init__(self, input_dim, latent_dim, hidden_dim = 100):
        super(vae, self).__init__()  
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

def omega_calculate(propensity_scores, ck, h, kernel = "gaussian"):
    """
    Calculate the kernel weight function omega based on the given propensity scores.

    Parameters:
    propensity_scores (torch.Tensor): The propensity scores.
    ck (float): The center of the kernel.
    h (float): The bandwidth of the kernel.
    kernel (str, optional): The type of kernel to use. Default is "gaussian".

    Returns:
    torch.Tensor: The weight function omega.

    """
    x = (propensity_scores - ck)/h

    omega = 0
    
    if(kernel == "uniform"): 
        omega = 0.5 * (torch.abs(x) <= 1)
        
    if(kernel == "epanechnikov"): 
        omega = 0.75 * (1 - x**2) * (torch.abs(x) <= 1)

    if(kernel == "gaussian"): 
        omega = 1/torch.sqrt(2 * torch.tensor(torch.pi)) * torch.exp(-x**2/2)
    
    return 1/h * omega

def local_balance_ipw_loss(propensity_scores, treatment, Z, ck, h, device):
    """
    Calculates the local balance IPW loss (Q1) for a given set of propensity scores, treatment assignment, covariates, and bandwidths.

    Parameters:
    propensity_scores (torch.Tensor): The propensity scores.
    treatment (torch.Tensor): The treatment assignments.
    Z (torch.Tensor): The covariates.
    ck (torch.Tensor): The center of the kernel.
    h (torch.Tensor): The bandwidth of the kernel.

    Returns:
    torch.Tensor: The local balance IPW loss.

    """
    
    ipw = treatment*propensity_scores + (1-treatment)*(1-propensity_scores) ## Calculate IPW weight
    K = len(ck)
    K_new = K
    loss = 0.0
    e = 1e-6
    
    for k in range(K):
        w = omega_calculate(propensity_scores, ck[k], h[k])
        w = torch.where((torch.abs(w) < e) & (w != 0), torch.full_like(w, e), w)
        W = w / ipw ## Add IPW
        
        q = (2*treatment - 1) * W
        q = q.reshape(-1,1)* Z
        qvecT = torch.sum(q, dim = 0)
        qvecT = qvecT.to(device)

        temp0 = w**2 
        temp0 = temp0.reshape(-1,1)* Z
        temp1 = torch.matmul(torch.transpose(Z, 0, 1), temp0)
        
        sigma = temp1 / (ck[k] * (1-ck[k]))
        sigma = sigma.to(device)
        sigma_inv = torch.linalg.pinv(sigma) ## Calculate sigma matrix
        
        A = torch.matmul(qvecT.double(), sigma_inv.double())
        loss += torch.matmul(A, qvecT.double())
        
        if torch.sum(w) == 0: ## If the sum of weights is 0, then the kernel is not used in the loss function
            K_new = K_new - 1   
        
    return loss/K_new

def penalty_loss(propensity_scores, treatment, ck, h):
    """
    Calculate the penalty loss term (Q2) for propensity scores.

    Parameters:
    propensity_scores (torch.Tensor): The propensity scores.
    treatment (torch.Tensor): The treatment values.
    ck (torch.Tensor): The center of the kernel.
    h (torch.Tensor): The bandwidth of the kernel.

    Returns:
    torch.Tensor: The calculated penalty loss term.
    """
    
    K = len(ck)
    K_new = K
    loss = 0.0	
    e = 1e-6

    for k in range(K):
        
        w = omega_calculate(propensity_scores, ck[k], h[k])
        w = torch.where((torch.abs(w) < e) & (w != 0), torch.full_like(w, e), w)
        part1 = w * (treatment - propensity_scores)**2
        numerator = torch.sum(part1, dim = 0)

        if torch.sum(w) == 0: ## Ensure that the denominator is not 0, and then the kernel is used in the loss function
            denominator = 1
            K_new = K_new - 1

        else:
            denominator = ck[k] * (1-ck[k]) * torch.sum(w)
            
        loss += numerator/denominator
    
    return loss/K_new

def lsd_cal(propensity_scores, treatment, Z, ck, h):
    """
    Calculate the LSD using propensity scores, treatment assignments, and covariates.

    Args:
    propensity_scores (Tensor): Propensity scores for the treatment.
    treatment (Tensor): Binary treatment assignment (1 for treatment, 0 for control).
    Z (Tensor): Matrix of covariates (including intercept as the first column).
    ck (Tensor): The center of the kernel.
    h (Tensor): The bandwidth of the kernel.

    Returns:
    Tuple: Maximum and mean absolute LSD values.
    """
    LSD = []
    K = len(ck)
    ipw = treatment*propensity_scores + (1-treatment)*(1-propensity_scores) 

    for k in range(K):
        w = omega_calculate(propensity_scores, ck[k], h[k])
        W = w / ipw ## Add IPW

        Z0 = Z[:,1:] ## remove the intercept to calculate LSD
        mu1 = torch.sum(treatment * W * Z0.T, dim=1) / torch.sum(treatment * W)
        mu0 = torch.sum((1 - treatment) * W * Z0.T, dim=1) / torch.sum((1 - treatment) * W)
        v1 = torch.sum(treatment * W * ((Z0 - mu1) ** 2).T,  dim=1) / torch.sum(treatment * W)
        v0 = torch.sum((1 - treatment) * W * ((Z0 - mu0) ** 2).T, dim=1) / torch.sum((1 - treatment) * W)
        ## effective sample size
        ess1 = (torch.sum(treatment * W)) ** 2 / torch.sum(treatment * W ** 2)
        ess0 = (torch.sum((1 - treatment) * W)) ** 2 / torch.sum((1 - treatment) * W ** 2)
        LSD.append(100 * (mu1 - mu0) / torch.sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0)))
    
    LSD = torch.stack(LSD)
    LSD_mean = torch.mean(torch.abs(LSD))
    LSD_max = torch.max(torch.abs(LSD))

    return LSD_max, LSD_mean