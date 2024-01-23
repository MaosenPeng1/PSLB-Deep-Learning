source('functions (data generation & evaluation & plots).R')
seed_file <- read.csv('sim_seed.csv', header = FALSE) ## seeds
seed_values <- seed_file[,1]

rho = 0.1 ## span
ck <- seq(0.05, 0.95, 0.01) ## selected local points

## generation of data for mis-specified and correctly specified data (true) for each seed
for (i in seq_along(seed_values)){
  
  seed = seed_values[i]
  DM = Kang_Schafer_Simulation(n = 5000, seeds = seed)
  Z = DM$Data[,3:6] 
  X = DM$Data[,7:10]
  treatment = DM$Data[,2]
  Y = DM$Data[,1]
  
  log.fit.true = glm(treatment~Z, family = binomial) ## true model
  log.fit.mis = glm(treatment~X, family = binomial) ## mis-specified model
  ps.true = log.fit.true$fitted.values ## initial propensity score used for adaptive bandwidth selection by default 
  ps.mis = log.fit.mis$fitted.values
  
  data = cbind(Y, "Tr" = treatment, Z, X)
  write.csv(data, paste0("KS5000_seed", seed, ".RData.csv"), row.names = FALSE)
  
  ## ck and adaptive bandwidth h
  h.true <- span_to_bandwidth(rho, ck, ps.true)
  input.true <- data.frame(ck=ck, h=h.true)
  write.csv(input.true, paste0("ck_h_true", seed, ".csv"), row.names = FALSE)
  
  h.mis <- span_to_bandwidth(rho, ck, ps.mis)
  input.mis <- data.frame(ck=ck, h=h.mis)
  write.csv(input.mis, paste0("ck_h_mis", seed, ".csv"), row.names = FALSE)
  
}