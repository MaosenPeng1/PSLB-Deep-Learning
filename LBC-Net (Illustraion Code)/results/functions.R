ipw <- function(Z, p){
  ## Purpose: inverse probability function
  ## Input: 
  ##    Z -- treatment variable
  ##    p -- propensity score
  ## Output:
  ##    single value or a vector under gaussian kernel
  weight <- Z/p + (1-Z)/(1-p)
  return(weight)
  
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

ATE_infer <-  function(Y, wt, Z) {
  ## Purpose: calculate estimated average treatment effect (ATE)
  ## Input: 
  ##    Y -- observed outcome
  ##    Z -- treatment variable
  ##    wt -- inverse probability weights
  ## Output:
  ##    a vector of estimated ATE
  ATE_case = sum(Z * wt * Y) / sum(Z * wt)
  ATE_control = sum((1-Z) * wt * Y) / sum((1-Z) * wt)
  
  ATE_after = ATE_case - ATE_control
  return(ATE_after)
  
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

GSD_wt <- function(X, Z, wt){
  ## Purpose: GSD estimation for methods that estimate weight directly
  ## Input: 
  ##    X -- covariate 
  ##    Z -- treatment variable
  ##    p -- propensity score
  ##    ck -- pre-specified points range from 0 to 1
  ##    h -- a vector of bandwidths 
  ##    kernel_function -- kernel function
  ## Output:
  ##    values of GSD for a covariate
  
  ## Estimating Global Standardized Differences (GSD)
  mu1 <- sum(Z * wt * X) / sum(Z * wt)
  mu0 <- sum((1-Z) * wt * X) / sum((1-Z) * wt)
  v1 <- sum(Z * wt * (X - mu1)^2) / sum(Z * wt)
  v0 <- sum((1-Z) * wt * (X - mu0)^2) / sum((1-Z) * wt)
  ess1 <- (sum(Z * wt))^2 / sum(Z * wt^2)
  ess0 <- (sum((1-Z) * wt))^2 / sum((1-Z) * wt^2)
  
  GSD <- 100 * (mu1 - mu0) / sqrt((ess1 * v1 + ess0 * v0) / (ess1 + ess0))
  
  return(GSD)
  
}

GSD <- function(X, Z, p){
  ## Purpose: GSD estimation
  ## Input: 
  ##    X -- covariate 
  ##    Z -- treatment variable
  ##    p -- propensity score
  ##    ck -- pre-specified points range from 0 to 1
  ##    h -- a vector of bandwidths 
  ##    kernel_function -- kernel function
  ## Output:
  ##    values of GSD for a covariate
  
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
  
  return(GSD)
  
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
  ##    a list containing LSD, mean of LSD, and GSD
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
  
  return(list(LSD=abs(LSD), LSD_mean = LSD_mean))
  
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

Col_bias_variance_rmse <-  function(Est, True) {
  ## Purpose: calculate estimated outcome measures
  ## Input: 
  ##    Y -- observed outcome
  ##    Z -- treatment variable
  ##    wt -- inverse probability weights
  ## Output:
  ##    a vector of estimated average outcome
  percent_bias <- 100*(mean(Est) - True)/True
  variance <- var(Est)
  rmse <- sqrt(mean((Est - True)^2))
  
  out <- data.frame(pbias = percent_bias, var = variance, rmse = rmse)
  
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

LSD_balance_plot <- function(ds){
  ## Purpose: Local balance plot
  ## Input: 
  ##    ds -- Data
  ## Output:
  ##    Local balance plot
  
  # Calculate column means
  f1_colmean <- function(mat){
    mean_matrix <- do.call(rbind, mat)
    return(colMeans(mean_matrix))
  }
  
  ds <- lapply(ds, f1_colmean)
  
  # Combine into a new data frame
  ds_new <- do.call(rbind, lapply(1:length(ds), function(i){
    data.frame(CK = 1:length(ds[[i]]),
               Mean = ds[[i]],
               Method = names(ds)[i])
  }))
  
  # Generate numeric labels for CK
  numeric_labels <- rep(sprintf("%.2f", seq(0.01, 0.99, by = 0.01)), length(ds))
  ds_new$CK <- numeric_labels
  
  # Define line types, shapes, and colors for each method
  line_types <- c("LOGISTIC" = "dotdash", "CBPS" = "dashed", "NN" = "dotted", "LBC-NET" = "solid")
  point_shapes <- c("LOGISTIC" = 15, "CBPS" = 17, "NN" = 16, "LBC-NET" = 1)
  custom_colors <- c("LOGISTIC" = "#d62728", "CBPS" = "#2ca02c", "NN" = "#1f77b4", "LBC-NET" = "#9467bd")
  
  # Plot with both points and boxplots having the same shape for consistency
  pt <- ggplot(ds_new, aes(x = CK, y = Mean, linetype = Method, shape = Method, color = Method)) +
    geom_line(size = 1.2) +  # Increased line width
    geom_point(size = 0.8) +  # Smaller point size for the line plot
    scale_x_discrete(breaks = c(0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 0.99), labels = c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)) +
    theme_bw(base_size = 20) + 
    theme(legend.position = "bottom",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 20),
          legend.text = element_text(size = 20),
          legend.title = element_text(size = 20)) +
    labs(x = "Propensity Score", y = "LSD(%)") +
    scale_linetype_manual(values = line_types) +
    scale_shape_manual(values = point_shapes) +
    scale_color_manual(values = custom_colors) +
    guides(
      shape = guide_legend(override.aes = list(size = 1.7))  # Custom size for the legend points
    )
  
  # Add boxplots for each method, sharing the same shapes and fill
  for (i in unique(ds_new$Method)){
    ds_sub <- subset(vdata, Method == i)
    pt <- pt + geom_boxplot(data = ds_sub, 
                            aes(x = CK, y = LSD, shape = Method, color = Method), outlier.shape = NA, 
                            width = 0.5, 
                            position = position_dodge(width = 0.5)) +
      geom_point(data = ds_sub, aes(x = CK, y = LSD, shape = Method), size = 0, position = position_dodge(width = 0.5)) +
      stat_boxplot(data = ds_sub,
                   aes(x = CK, y = LSD, color = Method, group = interaction(CK, Method)),
                   geom = "errorbar",
                   width = 1.5,
                   position = position_dodge(width = 0.5))
  }
  
  # Restrain y-axis to (0, 35)
  pt <- pt + scale_y_continuous(limits = c(0, 20))
  
  return(pt)
}

Model_based_estimates <- function(ps) {
  ## Purpose: Calculate the estimates for model-based methods
  ## Input: 
  ##    ps -- propensity score
  ## Output:
  ##    list of estimates
  
  # Initialize variables for GSD and LSD
  gsd <- rep(NA, p)
  L <- rep(NA, 99)
  Var_name <- names(X)
  TLSD <- setNames(lapply(Var_name, function(x) L), Var_name)
  
  ## ATE estimation (length of stay)
  wt <- ipw(Z, ps)  # Inverse probability weights
  ATE_los <- ATE_infer(los, wt, Z)
  
  ## ATE estimation (in_hospital_mortality)
  ATE_in_hospital_mortality <- ATE_infer(in_hospital_mortality, wt, Z)
  
  ## HR (survival)
  fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, data = ds, weights = wt, robust = TRUE)
  hr <- exp(coef(fit))  # HR is the exponentiated coefficient
  ci_lower <- exp(confint(fit)[, 1])  
  ci_upper <- exp(confint(fit)[, 2])
  
  ## GSD estimation
  GSD_res <- apply(X, 2, function(x) GSD(x, Z, ps)) 
  gsd <- unlist(lapply(GSD_res, function(x) abs(x)))
  
  ## LSD estimation
  LSD_res <- apply(X, 2, function(x) LSD(x, Z, ps, ck, h, gaussian_kernel)) 
  for (j in 1:p) {
    TLSD[[Var_name[j]]] <- LSD_res[[Var_name[j]]]$LSD
  }
  
  mean_LSD <- unlist(lapply(LSD_res, function(x) x$LSD_mean))
  mean_LSD <- mean(mean_LSD)
  
  ## Result summary
  res <- list(
    ATE_los = ATE_los,
    ATE_in_hospital_mortality = ATE_in_hospital_mortality,
    HR = hr,
    TLSD = TLSD,
    GSD = gsd,
    LSD = mean_LSD
  )
  
  return(res)
  
}

Weight_based_estimates <- function(wt, type = 1) {
  ## Purpose: Calculate the estimates for model-based methods
  ## Input: 
  ##    ps -- propensity score
  ##    type -- type of weights, 1 is ate weight, 2 is att and atc weights.
  ## Output:
  ##    list of estimates
  
  # Initialize variables for GSD
  gsd <- rep(NA, p)

  if (type == 1){
    
    ## ATE estimation (los)
    ATE_los <- ATE_infer(los, wt, Z)
    
    ## ATE estimation (in_hospital_mortality)
    ATE_in_hospital_mortality <- ATE_infer(in_hospital_mortality, wt, Z)
    
    ## HR (survival)
    ds_nonzero_weight <- ds[wt > 0, ]
    wt_nonzero <- wt[wt > 0]
    fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, 
                 data = ds_nonzero_weight, weights = wt_nonzero, robust = TRUE)
    hr <- exp(coef(fit))  # HR is the exponentiated coefficient
    ci_lower <- exp(confint(fit)[, 1])  
    ci_upper <- exp(confint(fit)[, 2])
    
    ## GSD estimation
    GSD_res <- apply(X, 2, function(x) GSD_wt(x, Z, wt)) 
    gsd <- unlist(lapply(GSD_res, function(x) abs(x)))
    
  } else if(type == 2){
    
    wt_att <- wt$wt_att
    wt_atc <- wt$wt_atc
    pi <- sum(Z)/length(Z)
    
    ## ATE estimation
    ATT <- ATE_infer(los, wt_att, Z)
    ATC <- ATE_infer(los, wt_atc, Z)
    ATE_los <- pi*ATT + (1-pi)*ATC
    
    ATT_in_hospital_mortality <- ATE_infer(in_hospital_mortality, wt_att, Z)
    ATC_in_hospital_mortality <- ATE_infer(in_hospital_mortality, wt_atc, Z)
    ATE_in_hospital_mortality <- pi*ATT_in_hospital_mortality + (1-pi)*ATC_in_hospital_mortality
    
    ## HR (survival)
    ds_nonzero_weight <- ds[wt_att > 0, ]
    wt_nonzero <- wt_att[wt_att > 0]
    fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, 
                 data = ds_nonzero_weight, weights = wt_nonzero, robust = TRUE)
    hr <- exp(coef(fit))  # HR is the exponentiated coefficient
    ci_lower <- exp(confint(fit)[, 1])  
    ci_upper <- exp(confint(fit)[, 2])
    
    ## GSD
    GSD_res <- apply(X, 2, function(x) GSD_wt(x, Z, wt_att)) 
    gsd <- unlist(lapply(GSD_res, function(x) abs(x)))
    
    
  }
  
  ## Result summary
  res <- list(
    ATE_los = ATE_los,
    ATE_in_hospital_mortality = ATE_in_hospital_mortality,
    HR = hr,
    GSD = gsd
  )
  
  return(res)
  
}

Model_based_ve <- function(ps_list) {
  ## Purpose: Calculate the standard deviation for model-based methods
  ## Input: 
  ##    ps -- list of propensity scores for all seeds
  ## Output:
  ##    list of sd(estimates)
  
  ATE_los <- ATE_in_hospital_mortality <- HR <- rep(NA, n)
  
  for (i in 1:n) {
    
    ds <- data_list[[i]]
    X <- ds[, 6:25]       # Covariates
    Z <- ds[, 5]          # Treatment indicator
    los <- ds[, 1]        # Length of stay
    in_hospital_mortality <- ds[, 2]  # in_hospital_mortality
    
    wt <- ipw(Z, ps_list[, i])
    
    # ATE estimation for length of stay
    ATE_los[i] <- ATE_infer(los, wt, Z)
    
    # ATE estimation for in_hospital_mortality
    ATE_in_hospital_mortality[i] <- ATE_infer(in_hospital_mortality, wt, Z)
    
    ## HR (survival)
    fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, data = ds, weights = wt, robust = TRUE)
    HR[i] <- exp(coef(fit))  # HR is the exponentiated coefficient
    
  }
  
  # Return the standard deviations of the estimates
  res <- list(
    los_sd = sd(ATE_los),
    in_hospital_mortality_sd = sd(ATE_in_hospital_mortality),
    HR_sd = sd(HR)
  )
  
  return(res)
  
}

Weight_based_ve <- function(wt_list, type = 1) {
  ## Purpose: Calculate the standard deviation for model-based methods
  ## Input: 
  ##    wt_list -- list of weights
  ##    type -- type of weights, 1 is ate weight, 2 is att and atc weights.
  ## Output:
  ##    list of sd(estimates)
    
    n <- length(s)
    ATE_los <- ATE_in_hospital_mortality <- HR <- rep(NA, n)
    if (type == 1){
    
      for (i in 1:n){
        
        ds <- data_list[[i]]
        X <- ds[,6:25]
        Z <- ds[,5]
        los <- ds[,1]
        in_hospital_mortality <- ds[,2]
        wt <- wt_list[,i]
        
        ## Y estimation
        ATE_los[i] <- ATE_infer(los, wt, Z)
        
        ## ATE estimation (in_hospital_mortality)
        ATE_in_hospital_mortality[i] <- ATE_infer(in_hospital_mortality, wt, Z)
        
        ## HR (survival)
        ds_nonzero_weight <- ds[wt > 0, ]
        wt_nonzero <- wt[wt > 0]
        fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, 
                     data = ds_nonzero_weight, weights = wt_nonzero, robust = TRUE)
        HR[i] <- exp(coef(fit))  # HR is the exponentiated coefficient

      }
    
  } else if(type == 2){
    
    wt_att <- wt_list$wt_att
    wt_atc <- wt_list$wt_atc

    for (i in 1:n){
      
      ds <- data_list[[i]]
      X <- ds[,6:25]
      Z <- ds[,5]
      los <- ds[,1]
      in_hospital_mortality <- ds[,2]
      pi <- sum(Z)/length(Z)
      wt <- wt_att[,i]
      
      ## ATE estimation
      ATT <- ATE_infer(los, wt, Z)
      ATC <- ATE_infer(los, wt, Z)
      ATE_los[i] <- pi*ATT + (1-pi)*ATC
      
      ## ATE estimation (in_hospital_mortality)
      ATE_in_hospital_mortality[i] <- ATE_infer(in_hospital_mortality, wt, Z)
      
      ## HR (survival)
      ds_nonzero_weight <- ds[wt > 0, ]
      wt_nonzero <- wt[wt > 0]
      fit <- coxph(Surv(survival_time_28_day, death_within_28_days) ~ Tr, 
                   data = ds_nonzero_weight, weights = wt_nonzero, robust = TRUE)
      HR[i] <- exp(coef(fit))  # HR is the exponentiated coefficient
      
    }
    
  }
  
  ## Result summary
  res <- list(
    los_sd = sd(ATE_los),
    in_hospital_mortality_sd = sd(ATE_in_hospital_mortality),
    HR_sd = sd(HR)
  )
  
  
  return(res)
  
}
