## load libraries
library(survival)
library(survminer)
library(dplyr)
library(pracma)
library(patchwork)
library(tidyr)
library(textables)
library(ggplot2)

## load data
source("functions.R")
file <- paste0("MIMIC_EPR.csv")
ds <- read.csv(file, header = TRUE)
X <- ds[,6:25]
Z <- ds[,5]
los <- ds[,1]
in_hospital_mortality <- ds[,2]
p <- ncol(X) 

# load ck_h
file_ck <- paste0("ck_h.csv")
ck_list <- read.csv(file_ck, header = TRUE)
ck <- ck_list$ck
h <- ck_list$h

## load propensity scores
ps_temp <- read.csv("ps_lbc_net.csv", header = FALSE)
ps_lbc_net <- ps_temp[,1]

## calculate estimates
lbc_net <- Model_based_estimates(ps_lbc_net)

# ----------------------------------
## Mirror Histogram
# ----------------------------------

p <- mirror_plot(ps_lbc_net, Z)
ggsave("mimic_mh_figure.pdf", plot = p, width = 8, height = 8, device = "pdf", family = "Times")

# ----------------------------------
## HL Plot
# ----------------------------------

ps_group <- cut(ps_lbc_net, breaks = breaks, include.lowest = TRUE)
group_data <- data.frame(ps0 = ps_lbc_net, Z, ps_group) %>%
  group_by(ps_group) %>%
  summarise(avg_ps = mean(ps0), prop_Z = mean(Z))

p <- group_data %>%
  ggplot(aes(x = avg_ps, y = prop_Z)) +
  geom_point(color = color) +
  geom_line(color = color, linetype = line_type) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Average Estimated Propensity Score", y = "Observed Proportion of Z = 1") +
  theme_bw(base_size = 20) +
  theme( panel.grid.major = element_blank(),  # Removes major grid lines
         panel.grid.minor = element_blank(),
         axis.title = element_text(size = 20),
         axis.text = element_text(size = 20),
         text = element_text(size = 20))
ggsave("mimic_hl_figure.pdf", plot = p, width = 8, height = 8, device = "pdf", family = "Times")


