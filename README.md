# Prpensity Score Local Balance (PSLB) Estimation Using Deep Learning
## Description
This project guides users how to implemented the PSLB deep learning method to estimate propensity scores introduced in 'A Deep Learning Approach to Nonparametric Propensity Score Estimation with Optimized Covariate Balance'. It consists of total 10 simulation settings and a real data analysis. All related files for each setting are contained in the zip file listed below,
- Kang and Schafer (2007) 
  - 5k (5000 sample size) [*]
    - True (correctly specified propensity score model)
    - Mis (mis-specified propensity score model)
  - 1k (1000 sample size)
    - True (correctly specified propensity score model)
    - Mis (mis-specified propensity score model)
- Hainmueller (2012)
  - Sd1 (correctly specified propensity score model)
  - Sd3 (mis-specified propensity score model)
- SSMR [*]
  - True (correctly specified propensity score model)
  - Mis (mis-specified propensity score model)
- EQLS (real data) [*]

*: The results of the file are shown in the main context of the article. 

Each file consists of two parts, estimation and evaluation. The estimation related to the deep learning are conducted using Python (.py) and others are implemented with R (.R). Besides the PSLB method, the file also includes logistic regression, CBPS, and deep learning method with BCE loss for estimation comparison. The figures and tables in the article are contained in the evaluation files. A seed file is in each simulation files so that the results can be reproductive.

## Installation and Setup
### Codes and Resources Used
* R: version -- 4.3.1
Editor: RStudio
Packages Required:
CBPS, dplyr, tidyr, ggplot2, knitr, kableExtra.

* Python: version -- 3.9.7
Editor: Visual Studio Code
Packages Required:
torch, numpy, pandas, sys, argparse.

* Sever: Linux
Editor: X-Win32
Purpose: run deep learning code parallel on sever. The bash file (.lsf) is generated by the file called 'lsf_generation*.py'.

## Data
The simulated data is generated using the "*Simulation.R" file for all seeds in 'sim_seed.csv'. 

The real data here is the European Quality of Life Survey (EQLS) study data, which is available from the UK Data Service https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7724\#!/access-data. Here we uploaded the cleaned data in file 'eqls_data.csv'. The EQLS is a survey conducted through questionnaires that encompasses adults from 35 European countries. We merged the data from the 2007 and 2011 iterations of the EQLS. Our focus is on determining whether conflicts between work and life balance, as self-reported by respondents, influence the mental well-being of the working adult population. The 2007 EQLS included 35,635 individuals, while the 2011 version had 43,636 participants. For ease of analysis, we omitted 21,800 participants from the 2007 dataset and 27,234 from the 2011 dataset due to incomplete information. Our final sample is comprised of 17,439 individuals experiencing conflicts in balancing work and life (either at work, at home, or both, referred to as cases) and 12,797 individuals with no or minor conflicts (referred to as controls). We assessed mental well-being as the outcome using the World Health Organization Five (WHO-5) well-being index, which ranges from 0 to 100. This index evaluates the respondent's emotional state over the preceding two weeks. The number of covariates is 72.

## Instructions
1. generate data using "*Simulation.R", you will get data and ck_h csv files. The ck_h contains the local points and adaptive bandwidth by default, one can specify or define there own local intervals. Note all R functions are included in the "Method.R" file.
2. implement different methods to the data to estimate the propensity scores. Logistic regression and CBPS methods can be applied by "Logistic and CBPS Model.R", which will output a R object saved to the local computer for further use. The BCE loss and PSLB-DL method are run on the sever using the "\*.py" files through the bash file generated from "lsf_generation*.py".
3. evaluate the global balance, local balance, and outcome estimation (mean outcome or average treatment effect) with "Figures and Tables.R". This code can output figures and tables presneted in the article. 

## LICENSE











  
