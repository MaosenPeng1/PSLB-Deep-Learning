Kang_Schafer_Simulation = function(n, beta = c(-1,0.5,-0.25,-0.1), 
                                   alpha = c(210,27.4,13.7,13.7,13.7),
                                   mu = rep(0,4), sd = diag(4), seeds){
  ## Purpose: data generation from KS simulation
  ## Input: 
  ##    n -- The number of observations to simulate.
  ##    beta -- A vector of coefficients for the propensity score model. Defaults to c(-1,0.5,-0.25,-0.1).
  ##    alpha -- A vector of coefficients for the outcome model. Defaults to c(210,27.4,13.7,13.7,13.7).
  ##    mu -- The mean vector for the multivariate normal distribution used to generate covariates. Defaults to a zero vector of length 4.
  ##    sd -- The covariance matrix for the multivariate normal distribution. Defaults to a 4x4 identity matrix.
  ##    seeds -- The seed value for random number generation to ensure reproducibility.
  ## Output:
  ##    a list of data and treatent effects
  require(MASS)
  set.seed(seeds)
  beta = matrix(beta, ncol = 1)
  alpha = matrix(alpha, ncol = 1)
  Z = mvrnorm(n, mu = mu, Sigma = sd) ## correctly specified model covariates
  Z1 = cbind(rep(1,n),Z)
  epsilon = rnorm(n, mean = 0, sd = 1)
  Y = Z1%*%alpha + epsilon
  ps = exp(Z%*%beta)/(1+exp(Z%*%beta))
  D = c()
  for(i in 1:n) {
    D[i] = rbinom(1, size = 1, prob = ps[i])
  }
  X1 = exp(Z[,1]/2)
  X2 = Z[,2]/(1+exp(Z[,1])) + 10
  X3 = (((Z[,1]*Z[,3])/25)+0.6)^3
  X4 = (Z[,2] + Z[,4] + 20)^2
  X = cbind(X1, X2, X3, X4) ## mis-specified model covariates
  XI = rnorm(n)
  out = cbind(Y, D, Z, X, ps, XI)
  colnames(out) = c("Y", "Tr", "Z1", "Z2", "Z3", "Z4", "X1", "X2", "X3", "X4","PS","XI")
  treatment = mean(out[which(D==1),1]) - mean(out[which(D==0),1])
  
  return(list(Data = out, Treat.effect = treatment))
  
}

uniform_kernel <- function(x){
  ## Purpose: uniform kernel function
  ## Input: 
  ##    x -- single value or a vector
  ## Output:
  ##    single value or a vector under uniform kernel
  K_x <- 0.5 * as.numeric(abs(x) <= 1)
  
  return(K_x)
  
}

gaussian_kernel <- function(x){
  ## Purpose: gaussian kernel function
  ## Input: 
  ##    x -- single value or a vector
  ## Output:
  ##    single value or a vector under gaussian kernel
  K_x <- 1/sqrt(2*pi) * exp(-x^2/2)
  
  return(K_x)
  
}

epanechnikov_kernel <- function(x){
  ## Purpose: epanechnikov kernel function
  ## Input: 
  ##    x -- single value or a vector
  ## Output:
  ##    single value or a vector under epanechnikov kernel
  K_x <- 0.75 * (1 - x^2) * as.numeric(abs(x) <= 1)
  
  return(K_x)
  
}

span_to_bandwidth <- function(rho, ck, p){
  ## Purpose: transform span to bandwidth
  ## Input: 
  ##    rho -- span
  ##    ck -- pre-specified points range from 0 to 1
  ##    p -- propensity score
  ## Output:
  ##    a vector of bandwidth for each ck
  N <- length(p)
  h <- numeric(length(ck))
  
  for (i in seq_along(ck)){
    
    d <- abs(ck[i] - p)
    d_sort <- sort(d)
    h[i] <- d_sort[ceiling(N * rho)]
    
  }
  
  return(h)
  
}

LSD <- function(X, Z, p, ck, h, kernel_function){
  ## Purpose: LSD and GSD estimation
  ## Input: 
  ##    X -- covariate 
  ##    Z -- treatment variable
  ##    p -- propensity score
  ##    ck -- pre-specified points range from 0 to 1
  ##    h -- a vector of bandwidths 
  ##    kernel_function -- kernel function
  ## Output:
  ##    a list containing absolute value of LSD, mean of LSD, and GSD
  N <- length(p)
  LSD <- numeric(length(ck))
  
  for (i in seq_along(ck)){
    
    ## Estimating Local Standardized Differences (LSD)
    w <- 1/h[i] * kernel_function( (ck[i] - p) / h[i])
    
    W <- w / (Z * p + (1-Z) * (1-p))
    
    mu1 <- sum(Z * W * X) / sum(Z * W)
    mu0 <- sum((1-Z) * W * X) / sum((1-Z) * W)
    v1 <- sum(Z * W * (X - mu1)^2) / sum(Z * W)
    v0 <- sum((1-Z) * W * (X - mu0)^2) / sum((1-Z) * W)
    
    ess1 <- (sum(Z * W))^2 / sum(Z * W^2)
    ess0 <- (sum((1-Z) * W))^2 / sum((1-Z) * W^2)
    LSD[i] <- 100 * (mu1 - mu0) / sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0))
    
    
  }
  
  LSD_mean <- mean(abs(LSD))
  
  ## Estimating Global Standardized Differences (GSD)
  w <- 1
  W <- w / (Z * p + (1-Z) * (1-p))
  mu1 <- sum(Z * W * X) / sum(Z * W)
  mu0 <- sum((1-Z) * W * X) / sum((1-Z) * W)
  v1 <- sum(Z * W * (X - mu1)^2) / sum(Z * W)
  v0 <- sum((1-Z) * W * (X - mu0)^2) / sum((1-Z) * W)
  ess1 <- (sum(Z * W))^2 / sum(Z * W^2)
  ess0 <- (sum((1-Z) * W))^2 / sum((1-Z) * W^2)
  
  GSD <- 100 * (mu1 - mu0) / sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0))
  
  return(list(LSD=abs(LSD), LSD_mean = LSD_mean, GSD=GSD))
  
}

ipw <- function(Z, p){
  ## Purpose: inverse probability weighting
  ## Input: 
  ##    Z -- treatment variable
  ##    ps -- propensity score
  ## Output:
  ##    a vector of inverse probability weights
  weight <- Z/p + (1-Z)/(1-p)
  
  return(weight)
  
}

ATE_infer= function(Y, wt, Z) {
  ## Purpose: calculate estimated ATE (treatment effect)
  ## Input: 
  ##    Y -- observed outcome
  ##    Z -- treatment variable
  ##    wt -- inverse probability weights
  ## Output:
  ##    single value of ATE
  ATE_case = sum(Z * wt * Y) / sum(Z * wt)
  ATE_control = sum((1-Z) * wt * Y) / sum((1-Z) * wt)
  ATE_after = ATE_case - ATE_control
  
  return(ATE_after)
  
}

Y_infer = function(Y, wt, Z) {
  ## Purpose: calculate estimated average outcome
  ## Input: 
  ##    Y -- observed outcome
  ##    Z -- treatment variable
  ##    wt -- inverse probability weights
  ## Output:
  ##    a vector of estimated average outcome
  mu_ipw = sum(Z * wt * Y) / sum(Z * wt)
  
  return(mu_ipw)
}

Col_bias_variance_rmse <-  function(Est, True) {
  ## Purpose: calculate estimated average outcome
  ## Input: 
  ##    Y -- observed outcome
  ##    Z -- treatment variable
  ##    wt -- inverse probability weights
  ## Output:
  ##    a vector of estimated average outcome
  percent_bias <- 100*(mean(Est, na.rm = T) - True)/True
  variance <- var(Est, na.rm = T)
  rmse <- sqrt(mean((Est - True)^2, na.rm = T))
  out <- c(percent_bias, rmse, variance)
  names(out) = c("bias", "RMSE", "Var")
  
  return(out)
  
}

mirror_plot <- function(ps, Z){
  ## Purpose: mirror histogram
  ## Input: 
  ##    ps -- propensity score
  ##    Z -- treatment variable
  ## Output:
  ##    mirror histogram 
  data <- data.frame(ps=ps, Z=Z)
  plot <- data %>%
    ggplot(aes(x = ps)) +
    geom_histogram(aes(y = after_stat(count)), 
                   fill = "white",
                   color = 'black',
                   data = ~ subset(., Z == 0), 
                   bins = 70) +
    geom_histogram(aes(y = -after_stat(count)), 
                   data = ~ subset(., Z == 1),
                   bins = 70,
                   fill = "white",
                   color = 'black') +
    geom_hline(yintercept = 0) +
    labs(x = "Propensity Score",
         y = "Frequency") +
    theme(panel.background = element_blank(),
          axis.line = element_line(colour = "black")) +
    scale_y_continuous(breaks = c(-50, 0, 50, 100, 150, 200),
                       label = c(50, 0, 50, 100, 150, 200)) +
    annotate("text", 
             label = "Z=0",
             x = 0.93,
             y = 100,
             size = unit(3, "pt")) +
    annotate("text", 
             label = "Z=1",
             x = 0.93,
             y = -60,
             size = unit(3, "pt"))
  
  return(plot)
  
}

LSD_plot_mis <- function(ds){
  ## Purpose: Figure 1
  ## Input: 
  ##    ds -- LSD output dataset
  ## Output:
  ##    Figure 1
  p <- length(ds)
  ds <- lapply(ds, function(x) sapply(x, function(y) colMeans(y)))
  ds <- do.call(rbind, lapply(1:p, function(i){
    
    n <- nrow(ds[[i]])
    data.frame(CK = 1:n,
               X1 = ds[[i]][,1],
               X2 = ds[[i]][,2],
               X3 = ds[[i]][,3],
               X4 = ds[[i]][,4],
               Method = names(ds)[i])
    
  }))
  
  numeric_labels <- rep(sprintf("%.2f", seq(0.05, 0.95, 0.01)), p)
  ds$CK <- numeric_labels
  
  ds_long <- pivot_longer(ds, cols = c(X1, X2, X3, X4), names_to = "Covariate", values_to = "LSD")
  ds_long$Method <- factor(ds_long$Method, levels = c("logistic", "cbps", "bce", "PSLBdl"))
  custom_colors <- c("logistic" = "#d62728", "cbps" = "#2ca02c", "bce" = "#1f77b4", "PSLBdl" = "#9467bd")
  
  # Create the plot
  p <- ggplot(ds_long, aes(x = CK, y = LSD, color = Method, group = interaction(Method, Covariate))) +
    geom_point(size = 0.8) +
    geom_line(size = 0.5) +
    scale_x_discrete(breaks = seq(0.05, 0.95, 0.15), labels = seq(0.05, 0.95, 0.15)) + 
    facet_wrap(~ Covariate, scales = "free_y") +
    theme_bw(base_size = 20) + 
    theme(legend.position = "bottom",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 20),
          legend.text = element_text(size = 20),
          legend.title = element_text(size = 20)) +
    labs(x = "Propensity Score", y = "LSD") +
    scale_color_manual(values = custom_colors)
  
  p <- p + geom_point(data = gsd[1, ], aes(x = CK, y = GSD, color = Method), shape = 17, size = 2) 
  # Add horizontal lines for GSD values
  for (i in 2:nrow(gsd)) {
    
    p <- p + geom_point(data = gsd[i, ], aes(x = CK, y = ScaledGSD, color = Method), shape = 17, size = 2) 
    
  }
  
  p <- p + scale_y_continuous(sec.axis = sec_axis(~ . / scaling_factor, name = "GSD"))
  
  return(p)
  
}

LSD_plot_true <- function(ds){
  ## Purpose: Figure 2
  ## Input: 
  ##    ds -- LSD output dataset
  ## Output:
  ##    Figure 2
  p <- length(ds)
  ds <- lapply(ds, function(x) sapply(x, function(y) colMeans(y)))
  ds <- do.call(rbind, lapply(1:p, function(i){
    
    n <- nrow(ds[[i]])
    data.frame(CK = 1:n,
               Z1 = ds[[i]][,1],
               Z2 = ds[[i]][,2],
               Z3 = ds[[i]][,3],
               Z4 = ds[[i]][,4],
               Method = names(ds)[i])
    
  }))
  
  numeric_labels <- rep(sprintf("%.2f", seq(0.05, 0.95, 0.01)), p)
  ds$CK <- numeric_labels
  
  ds_long <- pivot_longer(ds, cols = c(Z1, Z2, Z3, Z4), names_to = "Covariate", values_to = "LSD")
  ds_long$Method <- factor(ds_long$Method, levels = c("true PS", "logistic", "cbps", "bce", "PSLBdl"))
  custom_colors <- c("true PS" = "#ff7f0e", "logistic" = "#d62728", "cbps" = "#2ca02c", "bce" = "#1f77b4", "PSLBdl" = "#9467bd")
  
  # Create the plot
  p <- ggplot(ds_long, aes(x = CK, y = LSD, color = Method, group = interaction(Method, Covariate))) +
    geom_point(size = 0.8) +
    geom_line(size = 0.5) +
    scale_x_discrete(breaks = seq(0.05, 0.95, 0.15), labels = seq(0.05, 0.95, 0.15)) + 
    facet_wrap(~ Covariate, scales = "free_y") +
    theme_bw(base_size = 20) + 
    theme(legend.position = "bottom",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 20),
          legend.text = element_text(size = 20),
          legend.title = element_text(size = 20)) +
    labs(x = "Propensity Score", y = "LSD") +
    scale_color_manual(values = custom_colors)
  
  p <- p + geom_point(data = gsd[1, ], aes(x = CK, y = GSD, color = Method), shape = 17, size = 2) 
  # Add horizontal lines for GSD values
  for (i in 2:nrow(gsd)) {
    
    p <- p + geom_point(data = gsd[i, ], aes(x = CK, y = ScaledGSD, color = Method), shape = 17, size = 2) 
    
  }
  
  p <- p + scale_y_continuous(sec.axis = sec_axis(~ . / scaling_factor, name = "GSD"))
  
  return(p)
  
}









