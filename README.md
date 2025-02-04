# LBC-Net: A Propensity Score Estimation Method

## Description

This project provides a comprehensive guide to implementing LBC-Net for
estimating propensity scores, as introduced in "A Deep Learning Approach
to Nonparametric Propensity Score Estimation with Optimized Covariate
Balance." It includes two main components: a case study detailing data
preprocessing and model application, and simulation experiments based on
settings from Kang and Schafer (2007) and Li and Li (2021), showcasing
the model's performance. The outlines of the files in this project are
outlined below,

-   LBC-Net (Illustration Code)
-   Complete Project Code
    -   Case Study (MIMIC-EPR)
        -   Data Application
        -   Data Preprocessing
    -   Simulation
        -   Kang and Schafer (2007) [KS]
        -   Li and Li (2021)
            -   M1
            -   M2
            -   M3

The project provides the `LBC-Net (Illustration Code)` for implementing
the LBC-Net methods, using the MIMIC-EPR case study as an example, along
with some illustrative results. For a comprehensive replication of the
entire study presented in the paper, refer to the
`Complete Project Code`.

Within the `Complete Project Code zip` file, you will find detailed
scripts for data generation, preprocessing, and the complete application
study, which includes causal inference, variance estimation, and
time-varying effects. The `Simulation` file contains two simulation
settings. The first is based on the benchmark study by Kang and Schafer
(2007) with sample sizes of 300, 1,000, and 5,000. The second is based
on Li and Li (2021) and includes three scenarios: M1, which involves
good overlap, homogeneous causal effects, and 20 covariates; M2, which
features poor overlap, both homogeneous and heterogeneous causal
effects, and 20 covariates; and M3, which includes good overlap,
homogeneous causal effects, 84 covariates, and a sample size of 30,000.

Each file contains a `function.R` script that is used for all R code
specific to that file.

There is a `function.R` in each file, which are used for all R code
specific to this file. In each sub-study, in addition to LBC-Net, the
code also compares performance with model-based methods (Logistic
Regression, CBPS, NN) and weight-based methods (SBW, CBIPM).

The following sections provide step-by-step guidance for implementing
the `LBC-Net (Illustration Code)`, along with comprehensive instructions
for the `KS Simulation with a sample size of 5000` and the
`Data Application`.

## Installation and Setup

### Codes and Resources Used

-   R: version -- 4.3.1 Editor: RStudio

-   Python: version -- 3.11.3 Editor: Visual Studio Code Required
    Modules: torch, numpy, pandas, argparse, time.

-   Sever: High Performance Computing (HPC) - Seadragon Cluster (Linux
    system) Editor: X-Win32 Hardware Resources: Central Processing Unit
    (CPU) or Graphics Processing Unit (GPU) Purpose: To run deep
    learning code in parallel on the server. The bash files (.lsf) are
    generated using the `lsf_script*.py` file, which is also used to
    specify model parameters when applicable.

-   Data: MIMIC-IV (Version 3.0) Download:
    <https://physionet.org/content/mimiciv/3.0/>

## Data (Coded as MIMIC-EPR)

The case study design is thoroughly detailed in the paper. The
`data_prepocess.R` file provides the data cleaning process and a
descriptive analysis of the dataset, with the required datasets from the
MIMIC-IV database listed in `datasets.txt`. The cleaned dataset is saved
as `MIMIC_EPR.csv`. This study investigates the causal relationship
between EPR changes and sepsis outcomes among ICU patients. The final
sample consists of 5,564 individuals, with 2,656 in the high EPR change
group and 2,908 in the low EPR change group. The primary outcome is
28-day survival, while the secondary outcomes include in-hospital length
of stay and in-hospital mortality. The dataset includes 20 baseline
covariates, covering demographic information, laboratory results, risk
scores, medication use, and treatments.

## Instructions (`LBC-Net (Illustration Code)`)

1.  Training Details and Hyperparameter Input The `lbc_net.py` script
    contains the core code for estimation, utilizing functions defined
    in `functions_lbc_net.py`. Parameter inputs are specified in the
    bash shell script `lsf_script_lbc_net.py`. Below, the details of the
    training process in `lbc_net.py` are explained:

Data Preparation:

-   Data Loading: The dataset and adaptive bandwidths (`ck_h.csv`) are
    loaded using a specific random seed to ensure reproducibility. The
    `ck_h.csv` file is generated with a span of 0.15.

-   Normalization: Covariates are normalized, and an intercept term is
    added to prepare the data for model training.

-   Covariate and Treatment Indicator: Ensure that the indices of the
    covariates and the treatment indicator are correctly specified in
    the script.

Model Configuration:

-   Initial Parameters:

    -   Input Dimensions: Set as the number of covariates in the dataset
        plus one (for the intercept).

    -   Hidden Dimensions: Determined based on experience, balancing the
        number of covariates and the total sample size to optimize model
        complexity while avoiding overfitting. The default is set to
        100, but for simpler models (e.g., Kang and Schafer
        simulations), it is set to 10.

    -   Output Dimensions: Configured to align with the sample size for
        propensity score estimation.

    -   Learning Rate: Default value is 0.05, providing a balance
        between training efficiency and performance. Sensitivity
        analysis indicates minimal differences in model performance (in
        terms of covariate balance and loss function) for learning rates
        between 0.001 and 0.1.

    -   Max Epochs: Set to 20,000 for sufficient training. For BCE
        methods, 250 epochs are used as this is generally adequate for
        convergence. Larger values of learning rates and epochs risk
        overfitting, producing extreme results (e.g., 0 or 1 scores).

    -   LSD threshold: This parameter enables early stopping to improve
        efficiency. Training stops when the rolling average of the LSD
        values drops below this threshold, as numerical studies show
        diminishing improvements after a certain number of epochs. The
        default is set to 2.

    -   Blalance Lambda: This term regulates the trade-off between the
        balancing term and the calibration term in LBC-Net. It is set to
        1 for the entire study. - GPU: The script is designed to
        automatically detect whether a GPU is available for computation.
        CPU is enough for most study purpose.

Model Architecture:

-   Network Structure: The default model is a three-layer fully
    connected feed-forward neural network, incorporating batch
    normalization to improve learning dynamics and stability. The number
    of hidden layers can be adjusted using the `L` parameter in the
    model. However, a three-layer structure is generally sufficient for
    most propensity score estimation tasks.

-   VAE for Initial Weights: A Variational Autoencoder (VAE) is used to
    initialize the weights of the propensity score model, providing a
    more effective starting point for the learning process.

-   Customized Loss Functions:

    -   local_balance_ipw_loss: Corresponds to $Q1$ in the article,
        focusing on achieving local covariate balance.

    -   penalty_loss (referred to as calibration loss in the article):
        Corresponds to $Q2$ in the article, ensuring the calibration of
        the propensity score model.

2.  Bash Files Generation and Execution For server-based estimation
    workflows (e.g., those run on a cluster), bash files with input
    parameters can be generated to streamline the process. If the server
    is not available, the main script `lbc_net.py` can be executed
    directly on a local machine. The expected runtime for this
    illustrative example is less than 5 minutes, ensuring efficiency
    during testing or demonstration. Upon completion, the script outputs
    the propensity scores in the ps_lbc_net.csv file.

3.  Analysis and Evaluation Use the illustrative R script
    `results (illustration).R` to evaluate the performance of the
    estimated propensity scores. This includes calculating the Average
    Treatment Effect (ATE), assessing global balance using the Global
    Standardized Difference (GSD), examining local balance with the
    Local Standardized Difference (LSD), and visualizing the results
    through a mirror histogram and a Hosmer-Lemeshow plot.
    
Follow the embedded instructions provided within each script to execute the corresponding programs on either a cluster or a local machine. Depending on the script, the programs can be run in Python or R, as specified. Ensure that the necessary dependencies and environment configurations are in place before execution.

## Instructions (`Data Application`)

-   Methods

This section contains all methods used for comparison in the case study.
The implementations for NN and CBIPM are provided as Python scripts with
corresponding bash files for execution, while the remaining methods
(CBPS, logistic regression, and SBW) are implemented in R.
These methods output propensity scores and weights as `.csv` files for
further analysis.

-   Results

This section includes all results generated from the methods. The `.rds`
file contains additional results for the time-varying hazard ratio study
and variance estimation. A script named `results (*).R` provides all the
code needed to generate analysis results, tables, and figures presented
in the paper. Ensure all required packages are installed before running
the script.

Note: The asterisk * in `results (*).R` represents the corresponding table 
or figure number in the manuscript or supplementary materials. "T" denotes 
a table, while "S" denotes a figure. For example, "TS1" refers to Table S1, 
and "FS1" refers to Figure S1.

-   Time-varying Effect

This file visualizes the time-varying effects of the Cox model for both
the unweighted model and the weighted model using LBC-Net. The variance
of the coefficients for constructing confidence intervals is estimated
using a bootstrap approach. The results from LBC-Net can leverage the
same outputs generated during the variance estimation process detailed
in the next section.

-   Variance Estimation

To estimate the variance for each method, 500 bootstrap samples are
generated using the `bootstrap.R` script. Initially, all methods produce
their respective estimated propensity scores or weights. Subsequently,
the `res_ve.R` script calculates the variance for each method based on
the bootstrap samples.

## Instructions (`KS 5k`)

- Data Generation

The `Kang_Schafer_Simulation.R` script generates 500 simulation samples with a specified sample size of n=5000. The data generation process can be configured as correctly specified (model=true) or misspecified (model=mis). The span is set to 0.15 by default, but this value can be adjusted as needed. The script outputs the simulated data and ck_h csv files, as well as the true propensity scores saved in a matrix format (`ps_true_matrix.rds`). The `ck_h.csv` file contains the local points and adaptive bandwidths by default, but users can specify or define their own local intervals if required.

Update: Due to the .rds files exceeding GitHub's file size limit, only the illustration files have been uploaded. The complete .rds files can be shared through alternative means if required.

- Estimation

Similar to the previous sections, the estimated propensity scores and weights are stored in `.csv` files. The `res_sim.R` script processes these raw estimations to generate results, including percent bias, RMSE, and covariate balance, which are saved in the `ks5k*.rds` file. All .rds files, including this one, can be found in the `results (.rds)` file. Afterward, the `res_summary (*).R` script can be used to replicate the figures and tables presented in the paper.

Note: The asterisk * in `res_summary (*).R` represents the corresponding table or figure number in the manuscript or supplementary materials. "T" denotes a table, while "S" denotes a figure. For example, "TS1" refers to Table S1, and "FS1" refers to Figure S1.

- Sensitivity Analysis

To evaluate the robustness of our findings, we performed a sensitivity analysis examining different network structures and model specifications. Three files are included: `layer`, `lr`, and `rho`, representing the tuning parameters for the number of hidden layers, learning rate, and span, respectively. The results for these tuning parameters can be obtained similarly through the `Estimation` process described above. For `rho`, the script generates files with varying numbers of spans, saved as `ck_h_{seed}_{rho}.csv`, where each file corresponds to a specific value of `rho`. 

- Ablation Study

Similarly, process the scripts individually for the following variations: balance loss only (`lbc_net_bal_only.py`), without VAE initialization (`lbc_net_no_vae.py`), and BCE loss only (`lbc_net_bce.py`). After running these scripts, use `res_sim.R` to process the raw estimation results and then `res_summary (*).R` to generate the final results, including figures and tables.

- Moments

Use estimated propensity scores to produce results for second moment balance. 

Please ensure that each step is followed carefully to guarantee the
accurate reproduction of our study's results. Your attention to detail
and adherence to these instructions are greatly appreciated and
essential for a thorough review process.

## LICENSE

This project is licensed under the MIT License - see the LICENSE.txt
file for details.

## Reference

Kang, J. D. and Schafer, J. L. Demystifying double robustness: A
comparison of alternative strategies for estimating a population mean
from incomplete data. Statistical Science, 22(4):523--539, 2007

Li, Y. and Li, L. (2021). Propensity score analysis methods with
balancing constraints: a monte carlo study. Statistical Methods in
Medical Research, 30(4):1119--1142.
