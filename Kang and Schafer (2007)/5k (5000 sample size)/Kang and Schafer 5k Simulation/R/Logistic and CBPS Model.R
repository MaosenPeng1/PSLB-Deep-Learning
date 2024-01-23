library(CBPS)

seed.values <- read.csv('sim_seed.csv', header = FALSE)
s <- seed.values[, 1]

file <- paste0(path, "KS5000_seed", s, ".RData.csv")
data_list <- lapply(file, read.csv, header = TRUE)

N <- matrix(nrow = 5000, ncol = length(s))
obj <- list(true = list(logistic = N, CBPS = N), 
            mis = list(logistic = N, CBPS = N))
for (i in seq_along(s)){
  
  data <- data_list[[i]]
  ## logistic regression
  log.fit.true <- glm(Tr~Z1+Z2+Z3+Z4, data = data, family = 'binomial')
  obj$true$logistic[,i] <- log.fit.true$fitted.values
  log.fit.mis <- glm(Tr~X1+X2+X3+X4, data = data, family = 'binomial')
  obj$mis$logistic[,i] <- log.fit.mis$fitted.values
  
  ## CBPS
  cbps.fit.true <- CBPS(Tr~Z1+Z2+Z3+Z4, data = data, method = 'exact', ATT = 0, twostep = FALSE)
  obj$true$CBPS[,i] <- cbps.fit.true$fitted.values
  cbps.fit.mis <- CBPS(Tr~X1+X2+X3+X4, data = data, method = 'exact', ATT = 0, twostep = FALSE)
  obj$mis$CBPS[,i] <- cbps.fit.mis$fitted.values
  
}

KSOther <- obj
save(KSOther, file = 'Other_Methods.RData')