library(knitr)
library(kableExtra)
library(ggplot2)
library(dplyr)
library(tidyr)

source('Methods.R')

########################################################################
################ Individual Data Results ###############################
########################################################################

seed <- 1005 ## example seed
testds <- read.csv(paste0("KS5000_seed", seed, ".RData.csv"))
file_ck_h <- paste0(path, "ck_h_true", seed, ".csv")
ck <- file_ck_h$ck
h <- file_ck_h$h
ps <- read.csv('KS_ps_final_true1005.csv', header = FALSE) ## read estimated propensity score from PSLB-DL method

## Hosmer-Lemshow plot
ps_group <- cut(ps, breaks = 10, include.lowest = TRUE)
group_data <- data.frame(ps, Z = testds[,2], ps_group) %>%
  group_by(ps_group) %>%
  summarise(avg_ps = mean(ps), prop_Z = mean(Z)) 

p <- group_data %>%
  ggplot(aes(x = avg_ps, y = prop_Z)) + 
  geom_point() +
  geom_line() +
  labs(x = "Average Estimated Propensity Score",
       y = "Observed Proportion of Z = 1") +
  theme_bw()
p
ggsave("Hosmer-Lemshow Plot.pdf", plot = p, width = 10, height = 8, device = "pdf", family = "Times")

## Mirror plot
mirror_plot(ps, Z = testds[,2])

########################################################################
################ All Simulations Results ###############################
########################################################################

load('Other_Methods.RData')
seed.values <- read.csv('sim_seed.csv', header = FALSE)
s <- seed.values[, 1]
file <- paste0(path, "KS5000_seed", s, ".RData.csv")
data_list <- lapply(file, read.csv, header = TRUE)

################ Mis-specified propensity score mdoel ###################

file_ck <- paste0(path, "ck_h_mis", s, ".csv")
ck_list <- lapply(file_ck, read.csv, header = TRUE)

## load propensity score for PSLB-DL
file_ps <- paste0("KS_ps_final_mis", s, ".csv")
ps_list <- lapply(file_ps, read.csv, header = FALSE)
ps_dl <- do.call(cbind, ps_list)
ps_dl <- apply(ps_dl, 2, as.numeric)

## load propensity score for BCE
file_ps <- paste0("KS_ps_final_mis_bce", s, ".csv")
ps_list <- lapply(file_ps, read.csv, header = FALSE)
ps_bce <- do.call(cbind, ps_list)
ps_bce <- apply(ps_bce, 2, as.numeric)

## ATE and LSD calculation
n <- length(s)
p <- length(ck_list[[1]]$ck)
M <- rep(NA, n)
N <- list(global = matrix(NA, nrow = n, ncol = 4), local = M)
L <- matrix(nrow = n, ncol = p)
Lm <- list(X1 = L, X2 = L, X3 = L, X4 = L)

Y_mis <- list(logistic = M, cbps = M, bce = M, PSLBdl = M)
Balance <- list(logistic = N, cbps = N, bce = N, PSLBdl = N)

## Whole LSD values
TLSD <- list(logistic = Lm, cbps = Lm, bce = Lm, PSLBdl = Lm)

for(i in 1:n) {
  
  KS <- data_list[[i]]
  
  ## IPW
  wt_logistic <- ipw(Z = KS$Data[,2], KSOther$mis$logistic[,i])
  wt_cbps <- ipw(Z = KS$Data[,2], ps = KSOther$mis$CBPS[,i])
  wt_dl <- ipw(Z = KS$Data[,2], ps = ps_dl[,i])
  wt_bce <- ipw(Z = KS$Data[,2], ps = ps_bce[,i])
  
  ## ATE estimate
  Y_mis$logistic[i] <- Y_infer(KS$Data[,1], wt_logistic, KS$Data[,2])
  Y_mis$cbps[i] <- Y_infer(KS$Data[,1], wt_cbps, KS$Data[,2])
  Y_mis$PSLBdl[i] <- Y_infer(KS$Data[,1], wt_dl, KS$Data[,2])
  Y_mis$bce[i] <- Y_infer(KS$Data[,1], wt_bce, KS$Data[,2])
  
  ## balance 
  ck <- ck_list[[i]]$ck
  h <- ck_list[[i]]$h
  
  ## PSLB DL
  PSLBdl <- apply(KS$Data[,7:10], 2, function(x) LSD(x, KS$Data[,2], ps_dl[,i], 
                                                     ck, h, gaussian_kernel)) 
  TLSD$PSLBdl$X1[i,] <- PSLBdl$X1$LSD
  TLSD$PSLBdl$X2[i,] <- PSLBdl$X2$LSD
  TLSD$PSLBdl$X3[i,] <- PSLBdl$X3$LSD
  TLSD$PSLBdl$X4[i,] <- PSLBdl$X4$LSD
  
  mean_LSD <- unlist(lapply(PSLBdl, function(x) x$LSD_mean))
  Balance$PSLBdl$global[i,] <- unlist(lapply(PSLBdl, function(x) x$GSD))
  Balance$PSLBdl$local[i] <- mean(mean_LSD)
  
  ## BCE
  bce <- apply(KS$Data[,7:10], 2, function(x) LSD(x, KS$Data[,2], ps_bce[,i], 
                                                  ck, h, gaussian_kernel)) 
  TLSD$bce$X1[i,] <- bce$X1$LSD
  TLSD$bce$X2[i,] <- bce$X2$LSD
  TLSD$bce$X3[i,] <- bce$X3$LSD
  TLSD$bce$X4[i,] <- bce$X4$LSD
  
  mean_LSD <- unlist(lapply(bce, function(x) x$LSD_mean))
  Balance$bce$global[i,] <- unlist(lapply(bce, function(x) x$GSD))
  Balance$bce$local[i] <- mean(mean_LSD)
  
  ## logistic regression
  logistic <- apply(KS$Data[,7:10], 2, function(x) LSD(x, KS$Data[,2], KSOther$mis$logistic[,i], 
                                                       ck, h, gaussian_kernel)) 
  TLSD$logistic$X1[i,] <- logistic$X1$LSD
  TLSD$logistic$X2[i,] <- logistic$X2$LSD
  TLSD$logistic$X3[i,] <- logistic$X3$LSD
  TLSD$logistic$X4[i,] <- logistic$X4$LSD
  
  mean_LSD <- unlist(lapply(logistic, function(x) x$LSD_mean))
  Balance$logistic$global[i,] <- unlist(lapply(logistic, function(x) x$GSD))
  Balance$logistic$local[i] <- mean(mean_LSD)
  
  ## CBPS
  cbps <- apply(KS$Data[,7:10], 2, function(x) LSD(x, KS$Data[,2], KSOther$mis$CBPS[,i], 
                                                   ck, h, gaussian_kernel)) 
  TLSD$cbps$X1[i,] <- cbps$X1$LSD
  TLSD$cbps$X2[i,] <- cbps$X2$LSD
  TLSD$cbps$X3[i,] <- cbps$X3$LSD
  TLSD$cbps$X4[i,] <- cbps$X4$LSD
  
  mean_LSD <- unlist(lapply(cbps, function(x) x$LSD_mean))
  Balance$cbps$global[i,] <- unlist(lapply(cbps, function(x) x$GSD))
  Balance$cbps$local[i] <- mean(mean_LSD)
  
}
res_mis <- list(Y_mis = Y_mis, Balance = Balance, TLSD = TLSD)
saveRDS(res_mis, "KS5kmis_res.rds")

################ Figure 1 ###################

res <- readRDS("KS5kmis_res.rds")
Y_mis <- res$Y_mis
Balance <- res$Balance
TLSD <- res$TLSD

## GSD
gsd <- lapply(Balance, function(x) abs(colMeans(x$global)))
gsd <- stack(gsd)
names(gsd) <- c("GSD", "Method")
gsd$Covariate <- rep(c("X1", "X2", "X3", "X4"), times = length(gsd))
gsd$CK <- c(rep('0.88', 4), rep('0.90', 4), rep('0.92', 4), rep('0.94', 4))
gsd$Method <- factor(gsd$Method, levels = c("logistic", "cbps", "bce", "PSLBdl"))

scaling_factor <- 10 
gsd$ScaledGSD <- gsd$GSD * scaling_factor

ggsave("KS (Mis-spesified Model) Figure.pdf", plot = LSD_plot_mis(TLSD), width = 10, height = 8, device = "pdf", family = "Times")

################ Correctlty specified propensity score model ###################

file_ck <- paste0(path, "ck_h_true", s, ".csv")
ck_list <- lapply(file_ck, read.csv, header = TRUE)

file_ps <- paste0("KS_ps_final_true", s, ".csv")
ps_list <- lapply(file_ps, read.csv, header = FALSE)
ps_dl <- do.call(cbind, ps_list)
ps_dl <- apply(ps_dl, 2, as.numeric)

file_ps <- paste0("KS_ps_final_true_bce", s, ".csv")
ps_list <- lapply(file_ps, read.csv, header = FALSE)
ps_bce <- do.call(cbind, ps_list)
ps_bce <- apply(ps_bce, 2, as.numeric)

n <- length(s)
p <- length(ck_list[[1]]$ck)
M <- rep(NA, n)
N <- list(global = matrix(NA, nrow = n, ncol = 4), local = M)
L <- matrix(nrow = n, ncol = p)
Lm <- list(Z1 = L, Z2 = L, Z3 = L, Z4 = L)

Y_true <- list(true = M, logistic = M, cbps = M, bce = M, PSLBdl = M)
Balance <- list(true = N, logistic = N, cbps = N, bce = N, PSLBdl = N)

## Whole LSD values
TLSD <- list(true = Lm, logistic = Lm, cbps = Lm, bce = Lm, PSLBdl = Lm)

for(i in 1:n) {
  
  KS <- data_list[[i]]
  
  ## IPW
  wt_true_model <- ipw(Z = KS$Data[,2], ps = KS$Data[,11])
  wt_logistic <- ipw(Z = KS$Data[,2], KSOther$true$logistic[,i])
  wt_cbps <- ipw(Z = KS$Data[,2], ps = KSOther$true$CBPS[,i])
  wt_dl <- ipw(Z = KS$Data[,2], ps = ps_dl[,i])
  wt_bce <- ipw(Z = KS$Data[,2], ps = ps_bce[,i])
  
  ## ATE estimate
  Y_true$true[i] <- Y_infer(KS$Data[,1], wt_true_model, KS$Data[,2])
  Y_true$logistic[i] <- Y_infer(KS$Data[,1], wt_logistic, KS$Data[,2])
  Y_true$cbps[i] <- Y_infer(KS$Data[,1], wt_cbps, KS$Data[,2])
  Y_true$PSLBdl[i] <- Y_infer(KS$Data[,1], wt_dl, KS$Data[,2])
  Y_true$bce[i] <- Y_infer(KS$Data[,1], wt_bce, KS$Data[,2])
  
  ## balance 
  ck <- ck_list[[i]]$ck
  h <- ck_list[[i]]$h
  
  ## true model
  true_model <- apply(KS$Data[,3:6], 2, function(x) LSD(x, KS$Data[,2], KS$Data[,11], 
                                                        ck, h, gaussian_kernel)) 
  TLSD$true$Z1[i,] <- true_model$Z1$LSD
  TLSD$true$Z2[i,] <- true_model$Z2$LSD
  TLSD$true$Z3[i,] <- true_model$Z3$LSD
  TLSD$true$Z4[i,] <- true_model$Z4$LSD
  
  mean_LSD <- unlist(lapply(true_model, function(x) x$LSD_mean))
  Balance$true$global[i, ] <- unlist(lapply(true_model, function(x) x$GSD))
  Balance$true$local[i] <- mean(mean_LSD)
  
  ## PSLB DL
  PSLBdl <- apply(KS$Data[,3:6], 2, function(x) LSD(x, KS$Data[,2], ps_dl[,i], 
                                                    ck, h, gaussian_kernel)) 
  TLSD$PSLBdl$Z1[i,] <- PSLBdl$Z1$LSD
  TLSD$PSLBdl$Z2[i,] <- PSLBdl$Z2$LSD
  TLSD$PSLBdl$Z3[i,] <- PSLBdl$Z3$LSD
  TLSD$PSLBdl$Z4[i,] <- PSLBdl$Z4$LSD
  
  mean_LSD <- unlist(lapply(PSLBdl, function(x) x$LSD_mean))
  Balance$PSLBdl$global[i,] <- unlist(lapply(PSLBdl, function(x) x$GSD))
  Balance$PSLBdl$local[i] <- mean(mean_LSD)
  
  ## BCE
  bce <- apply(KS$Data[,3:6], 2, function(x) LSD(x, KS$Data[,2], ps_bce[,i], 
                                                 ck, h, gaussian_kernel)) 
  TLSD$bce$Z1[i,] <- bce$Z1$LSD
  TLSD$bce$Z2[i,] <- bce$Z2$LSD
  TLSD$bce$Z3[i,] <- bce$Z3$LSD
  TLSD$bce$Z4[i,] <- bce$Z4$LSD
  
  mean_LSD <- unlist(lapply(bce, function(x) x$LSD_mean))
  Balance$bce$global[i,] <- unlist(lapply(bce, function(x) x$GSD))
  Balance$bce$local[i] <- mean(mean_LSD)
  
  ## logistic regression
  logistic <- apply(KS$Data[,3:6], 2, function(x) LSD(x, KS$Data[,2], KSOther$true$logistic[,i], 
                                                      ck, h, gaussian_kernel)) 
  TLSD$logistic$Z1[i,] <- logistic$Z1$LSD
  TLSD$logistic$Z2[i,] <- logistic$Z2$LSD
  TLSD$logistic$Z3[i,] <- logistic$Z3$LSD
  TLSD$logistic$Z4[i,] <- logistic$Z4$LSD
  
  mean_LSD <- unlist(lapply(logistic, function(x) x$LSD_mean))
  Balance$logistic$global[i,] <- unlist(lapply(logistic, function(x) x$GSD))
  Balance$logistic$local[i] <- mean(mean_LSD)
  
  ## CBPS
  cbps <- apply(KS$Data[,3:6], 2, function(x) LSD(x, KS$Data[,2], KSOther$true$CBPS[,i], 
                                                  ck, h, gaussian_kernel)) 
  TLSD$cbps$Z1[i,] <- cbps$Z1$LSD
  TLSD$cbps$Z2[i,] <- cbps$Z2$LSD
  TLSD$cbps$Z3[i,] <- cbps$Z3$LSD
  TLSD$cbps$Z4[i,] <- cbps$Z4$LSD
  
  mean_LSD <- unlist(lapply(cbps, function(x) x$LSD_mean))
  Balance$cbps$global[i,] <- unlist(lapply(cbps, function(x) x$GSD))
  Balance$cbps$local[i] <- mean(mean_LSD)
  
}
res_true <- list(Y_true = Y_true, Balance = Balance, TLSD = TLSD)
saveRDS(res_true, "KS5ktrue_res.rds")

################ Figure 2 ###################

res <- readRDS("KS5ktrue_res.rds")
Y_true <- res$Y_true
Balance <- res$Balance
TLSD <- res$TLSD
names(TLSD)[1] <- "true PS"
names(Balance)[1] <- "true PS"

## GSD
gsd <- lapply(Balance, function(x) abs(colMeans(x$global)))
gsd <- stack(gsd)
names(gsd) <- c("GSD", "Method")
gsd$Covariate <- rep(c("Z1", "Z2", "Z3", "Z4"), times = 5)
gsd$CK <- c(rep('0.86', 4), rep('0.88', 4), rep('0.90', 4), rep('0.92', 4), rep('0.94', 4))
gsd$Method <- factor(gsd$Method, levels = c("true PS", "logistic", "cbps", "bce", "PSLBdl"))
scaling_factor <- 10  
gsd$ScaledGSD <- gsd$GSD * scaling_factor

ggsave("KS (Correctly Specified Model) Figure.pdf", plot = LSD_plot_true(TLSD), width = 10, height = 8, device = "pdf", family = "Times")

################ Table 1 ###################

A <- 210 ## true average outcome
Y_summary <- rbind(true= Col_bias_variance_rmse(Y_true$true, A), 'true logistic' = Col_bias_variance_rmse(Y_true$logistic, A),
                   'true CBPS' = Col_bias_variance_rmse(Y_true$cbps, A), 'true bce' = Col_bias_variance_rmse(Y_true$bce, A), 'true PSLB DL' = Col_bias_variance_rmse(Y_true$PSLBdl, A),
                   'mis logistic' = Col_bias_variance_rmse(Y_mis$logistic, A), 'mis CBPS' = Col_bias_variance_rmse(Y_mis$cbps, A), 'mis bce' = Col_bias_variance_rmse(Y_mis$bce, A),
                   'mis PSLB DL' = Col_bias_variance_rmse(Y_mis$PSLBdl, A)) %>% round(4)
Y_summary <- as.data.frame(Y_summary)
Y_summary$Method <- c("True PS", rep(c("Logistic", "CBPS", "BCE", "PSLBDL"), 2))
Y_summary$Type <- c(rep("Correctly Specified Model", 5), rep("Mis-specified Model", 4))

#dput(Y_summary)
data_wide <- Y_summary %>%
  pivot_wider(names_from = Type, values_from = c(bias, RMSE, Var)) %>%
  dplyr::select(Method, `bias_Correctly Specified Model`, `RMSE_Correctly Specified Model`, `Var_Correctly Specified Model`, `bias_Mis-specified Model`, `RMSE_Mis-specified Model`, `Var_Mis-specified Model`) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), '-', .)))

table <- kable(data_wide, format = "latex", booktabs = TRUE, align = 'c', 
               col.names = c("", rep("", ncol(data_wide)-1))) %>%
  kable_styling(latex_options = c("striped", "scale_down")) %>%
  add_header_above(c("Method" = 1, "%Bias" = 1, "RMSE" = 1, "Var" = 1, "%Bias" = 1, "RMSE" = 1, "Var" = 1)) %>%
  add_header_above(c(" " = 1, "Correctly Specified Model" = 3, "Mis-spesified Model" = 3)) 
table_latex <- as.character(table)
table_latex <- gsub("\\\\midrule\\n", "\\\\midrule\\n\\noalign{}", table_latex)
writeLines(table_latex, "KS_Ytable.tex")









