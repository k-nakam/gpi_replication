#######################################################################
# This is the R script to visualize the results in the original article
# Procedure:
# 1. Run all the simulation separately for 
    # (1) OLS
    # (2) GPI (create)
    # (3) GPI (reuse)
    # (4) BERT
# 2. For each, you should have .csv file. You need to specify each file
# 3. Run the entire code
#######################################################################
rm(list = ls())
# load the required package
library(tidyverse)
library(ggpubr)
library(ggh4x)
library(scales)

# load results
## (1) OLS
res_ols <- read.csv("results/result_ols.csv")
## (2) GPI (create)
res_create <- read.csv("results/result_create.csv")
## (3) GPI (reuse)
res_reuse <- read.csv("results/result_reuse.csv")
## (4) BERT
res_bert <- read.csv("results/result_bert.csv")
## (5) GPI (no interaction)
res_noint <- read.csv("results/result_noint.csv")

# helper function
calculate_bias <- function(ate, truth){
  # calculate absolute bias
  truth <- mean(truth)
  bias <- mean(abs(ate - truth))
  return(bias)
}

calculate_coverage <- function(ate, se, truth, bias_corrected = FALSE){
  truth <- mean(truth)
  if (bias_corrected){
    bias <- calculate_bias(ate, truth)
    ate <- ate - bias
  }
  lower <- ate - 1.96 * se; upper <- ate + 1.96 * se
  covered <- (lower < truth) & (upper > truth)
  return(mean(covered))
}

calculate_cilength <- function(se){
  return(mean(2*1.96*se))
}

calculate_mse <- function(ate, truth){
  truth <- mean(truth)
  mse <- sqrt(mean((ate - truth)^2))
  return(mse)
}

##########################################################################################
## Figure 3: Performance of Five Estimators under Different Confounding Scenarios
##########################################################################################
# OLS result
res_diff_in_mean <- res_ols %>%
  # In DGP, oracle is 11.795 for first confounder and 10.00 for second confounder
  mutate(oracle = ifelse(confounding == "2", 11.795, 10.00)) %>%
  group_by(confounding, strength) %>%
  summarise(coverage = calculate_coverage(ols, se.ols, truth = oracle),
            bias = calculate_bias(ols, oracle),
            mse = calculate_mse(ols, oracle),
            sd_est = sd(ols),
            avg_cilength = calculate_cilength(se.ols),
            time = mean(time.ols)) %>%
  mutate(method = "Diff-in-Means") %>%
  ungroup()

# GPI (create)
res_TarNet <- res_create %>%
  mutate(oracle = ifelse(confounding == "2", 11.795, 10.00)) %>%
  group_by(confounding, strength) %>%
  summarise(coverage = calculate_coverage(TarNet, se.TarNet, truth = oracle),
            bias = calculate_bias(TarNet, oracle),
            mse = calculate_mse(TarNet, oracle),
            sd_est = sd(TarNet),
            avg_cilength = calculate_cilength(se.TarNet),
            time = mean(time.TarNet)) %>%
  mutate(method = "GPI (new texts)") %>%
  ungroup()

# GPI (reuse)
res_TarNet2 <- res_reuse %>%
  mutate(oracle = ifelse(confounding == "2", 11.795, 10.00)) %>%
  group_by(confounding, strength) %>%
  summarise(coverage = calculate_coverage(TarNet, se.TarNet, truth = oracle),
            bias = calculate_bias(TarNet, oracle),
            mse = calculate_mse(TarNet, oracle),
            sd_est = sd(TarNet),
            avg_cilength = calculate_cilength(se.TarNet),
            time = mean(time.TarNet)) %>%
  mutate(method = "GPI (text reuse)") %>%
  ungroup()

# BERT (outcome model)
res_pryzant <- res_bert %>%
  mutate(oracle = ifelse(confounding == "2", 11.795, 10.00)) %>%
  group_by(confounding, strength) %>%
  summarise(
    coverage = calculate_coverage(pryzant, se.pryzant, truth = oracle),
    bias = calculate_bias(pryzant, oracle),
    mse = calculate_mse(pryzant, oracle),
    sd_est = sd(pryzant),
    avg_cilength = calculate_cilength(se.pryzant),
    coverage = calculate_coverage(pryzant, se.pryzant, truth = oracle),
    time = mean(time.pryzant)) %>%
  mutate(method = "Outcome Model with BERT") %>%
  ungroup()

# BERT (DML)
res_gui <- res_bert %>%
  mutate(oracle = ifelse(confounding == "2", 11.795, 10.00)) %>%
  group_by(confounding, strength) %>%
  slice_head(n = 200) %>%
  summarise(
    coverage = calculate_coverage(gui, se.gui, truth = oracle),
    bias = calculate_bias(gui, oracle),
    mse = calculate_mse(gui, oracle),
    sd_est = sd(gui),
    avg_cilength = calculate_cilength(se.gui),
    coverage = calculate_coverage(gui, se.gui, truth = oracle),
    time = mean(time.gui)) %>%
  mutate(method = "DML with BERT") %>%
  ungroup()

# check results numerically
rbind(res_diff_in_mean, res_TarNet, res_TarNet2, res_pryzant, res_gui) %>%
  filter(confounding == "2") %>%
  rename("RMSE" = "mse") %>%
  select(-c(sd_est, avg_cilength)) %>%
  mutate(
    strength = factor(strength, levels = c("weak", "moderate", "strong")),
    method = factor(method, levels = c("GPI (new texts)", "GPI (text reuse)", "Diff-in-Means", "Outcome Model with BERT", "DML with BERT"))
  ) %>%
  rename(c("Bias" = "bias",
           "Method" = "method",
           "95% CI Coverage" = "coverage",
           "Avg. Time (sec.)" = "time"))

# A tibble: 15 × 7
#confounding strength `95% CI Coverage`    Bias    RMSE `Avg. Time (sec.)` Method                 
#<int> <fct>                <dbl>   <dbl>   <dbl>              <dbl> <fct>                  
#  1           2 moderate             0       7.95    7.95            0.000908 Diff-in-Means          
#  2           2 strong               0      86.0    86.0             0.000887 Diff-in-Means          
#  3           2 weak                 0       3.61    3.61            0.00137  Diff-in-Means          
#  4           2 moderate             0.945   2.03    2.72           43.6      GPI (new texts)        
#  5           2 strong               0.88   27.9    36.9            55.3      GPI (new texts)        
#  6           2 weak                 0.935   0.796   1.06           42.1      GPI (new texts)        
#  7           2 moderate             0.93    1.78    2.36           62.9      GPI (text reuse)       
#  8           2 strong               0.92   26.2    36.0            74.3      GPI (text reuse)       
#  9           2 weak                 0.92    0.749   0.982          54.1      GPI (text reuse)       
# 10           2 moderate             0.065   3.40    4.06          657.       Outcome Model with BERT
# 11           2 strong               0     112.    114.           2731.       Outcome Model with BERT
# 12           2 weak                 0.115   1.18    1.53          347.       Outcome Model with BERT
# 13           2 moderate             0.92    2.83   18.7           682.       DML with BERT          
# 14           2 strong               0.26  459.    917.           2756.       DML with BERT          
# 15           2 weak                 0.925   2.17    9.90          373.       DML with BERT          


conf <- rbind(res_diff_in_mean, res_TarNet, res_TarNet2,  res_pryzant, res_gui) %>%
  rename("RMSE" = "mse") %>%
  select(-c(sd_est, avg_cilength)) %>%
  mutate(
    strength = factor(strength, levels = c("weak", "moderate", "strong")),
    method = factor(method, levels = c("GPI (new texts)", "GPI (text reuse)",  "Diff-in-Means", "Outcome Model with BERT", "DML with BERT"))
  ) %>%
  rename(c("Absolute Bias" = "bias",
           "Method" = "method",
           "95% CI Coverage" = "coverage",
           "Avg. Runtime (sec.)" = "time")) %>%
  pivot_longer(!c(confounding, strength, Method)) %>%
  mutate(name = factor(name, levels = c("Absolute Bias", "RMSE", "95% CI Coverage", "Avg. Runtime (sec.)"))) %>%
  filter(Method != "Diff-in-Means")

bias_conf <- conf %>%
  mutate(confounding = ifelse(confounding == 2, "w/ separability", "w/o separability")) %>%
  filter(name == "Absolute Bias") %>%
  ggplot(aes(x = strength, y = value, color = Method, group = Method)) +  
  geom_point(aes(shape = Method)) +  # Different symbols for each method
  geom_line(aes(linetype = Method)) + # Different line types for each method
  facet_grid(confounding ~ name) +
  xlab("") + 
  ylab("") + 
  theme_classic() + 
  theme(legend.position = "bottom") +
  scale_y_continuous(
    trans = "log10", 
    breaks = c(1, 10, 100, 1000), 
    labels = c("0", "10", "100", "1000")
  ) +
  scale_color_manual(values = c("GPI (new texts)" = "red", 
                                "GPI (text reuse)" = "yellow4",  
                                "Diff-in-Means" = "gray69", 
                                "Outcome Model with BERT" = "blue", 
                                "DML with BERT" = "purple")) +
  scale_shape_manual(values = c(16, 12, 17, 18, 15)) +  # Set different symbols for each method
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dotted", "dotdash")) + # Set different line types
  theme(legend.title = element_blank()) +
  theme(strip.text.y = element_blank())

rmse_conf <- conf %>%
  mutate(confounding = ifelse(confounding == 2, "w/ separability", "w/o separability")) %>%
  filter(name == "RMSE") %>%
  ggplot(aes(x = strength, y = value, color = Method, group = Method)) +  
  geom_point(aes(shape = Method)) +  # Different symbols for each method
  geom_line(aes(linetype = Method)) + # Different line types for each method
  facet_grid(confounding ~ name) +
  xlab("") + 
  ylab("") + 
  theme_classic() + 
  theme(legend.position = "bottom") + 
  scale_y_continuous(
    trans = "log10", 
    breaks = c(1, 10, 100, 1000), 
    labels = c("0", "10", "100", "1000")
  ) +
  scale_shape_manual(values = c(16, 12, 17, 18, 15)) +  # Set different symbols for each method
  scale_color_manual(values = c("GPI (new texts)" = "red", 
                                "GPI (text reuse)" = "yellow4",  
                                "Diff-in-Means" = "gray69", 
                                "Outcome Model with BERT" = "blue", 
                                "DML with BERT" = "purple")) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dotted", "dotdash")) + # Set different line types
  theme(legend.title = element_blank()) +
  theme(strip.text.y = element_blank())

ci_conf <- conf %>%
  mutate(confounding = ifelse(confounding == 2, "w/ separability", "w/o separability")) %>%
  filter(name == "95% CI Coverage") %>%
  ggplot(aes(x = strength, y = value, color = Method, group = Method)) +  
  geom_point(aes(shape = Method)) +  # Different symbols for each method
  geom_line(aes(linetype = Method)) + # Different line types for each method
  facet_grid(confounding ~ name) +
  xlab("") + 
  ylab("") + 
  theme_classic() + 
  scale_y_continuous(limits = function(x) {
    if(all(x == 0)) {
      return(c(0, 1))
    } else if (min(x) >= 0 & max(x) <= 1) {
      return(c(0, 1))
    } else {
      return(x)
    }
  }) + 
  geom_hline(data = data.frame(name = factor("95% CI Coverage"), value = c(0.95)),
             aes(yintercept = value), color = "black", linetype = "dotted") + 
  theme(legend.position = "bottom") + 
  scale_shape_manual(values = c(16, 12, 17, 18, 15)) +  # Set different symbols for each method
  scale_color_manual(values = c("GPI (new texts)" = "red", 
                                "GPI (text reuse)" = "yellow4",  
                                "Diff-in-Means" = "gray69", 
                                "Outcome Model with BERT" = "blue", 
                                "DML with BERT" = "purple")) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dotted", "dotdash")) + # Set different line types
  theme(legend.title = element_blank())  +
  theme(strip.text.y = element_blank())

time_conf <- conf %>%
  mutate(confounding = ifelse(confounding == 2, "Under separability", "No separability")) %>%
  filter(name == "Avg. Runtime (sec.)") %>%
  mutate(confounding = factor(confounding, levels = c("Under separability", "No separability"))) %>%
  ggplot(aes(x = strength, y = value, color = Method, group = Method)) +  
  geom_point(aes(shape = Method)) +  # Different symbols for each method
  geom_line(aes(linetype = Method)) + # Different line types for each method
  facet_grid(confounding ~ name) +
  xlab("") + 
  ylab("") + 
  theme_classic() + 
  theme(legend.position = "bottom") + 
  scale_y_continuous(
    trans = "log10", 
    breaks = c(1, 10, 30, 100, 300, 1000, 3000), 
    labels = c("0", "10", "30", "100", "300", "1000", "3000")
  ) +
  scale_shape_manual(values = c(16, 12, 17, 18, 15)) +  # Set different symbols for each method
  scale_color_manual(values = c("GPI (new texts)" = "red", 
                                "GPI (text reuse)" = "yellow4",  
                                "Diff-in-Means" = "gray69", 
                                "Outcome Model with BERT" = "blue", 
                                "DML with BERT" = "purple")) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "dotted", "dotdash")) + # Set different line types
  theme(legend.title = element_blank())

ggarrange(bias_conf, rmse_conf, ci_conf, time_conf,
          nrow = 1, common.legend = TRUE, legend = "bottom",
          widths = c(1,1,1,1.1))


##########################################################################################
## Figure 4: Propensity Score Plot
##########################################################################################
ps_save <- read.csv("results/ps_save.csv")

ps_save %>%
  mutate(confounding = factor(confounding, levels = c("weak", "moderate", "strong")),
         method = factor(method, levels = c("GPI (new texts)", "GPI (text reuse)", "DML with BERT")),
         type = factor(type, levels = c("Under separability", "No separability")),
         Treatment = factor(Treatment, levels = c("Treatment Group", "Control Group"))) %>%
  group_by(confounding, type, method, Treatment) %>%
  ggplot(aes(x = tpreds, y = confounding, fill = Treatment)) +
  xlim(c(0, 1.0)) +
  geom_boxplot(
    outlier.size = 0.5, coef = 10
  ) +
  facet_nested(type ~ method, scales = "free") +
  theme_classic() +
  labs(x = "Propensity Score Estimates", y = "") +
  theme(legend.position = "bottom") + 
  theme(legend.title = element_blank())

##########################################################################################
## Figure 5: Results with Varying N (finite sample performance)
##########################################################################################
res_noint %>%
  mutate(
    type = case_when(type == "create" ~ "GPI (new texts)",
                     type == "reuse" ~ "GPI (text reuse)")
  ) %>%
  group_by(N, type) %>%
  summarise(Coverage = calculate_coverage(TarNet, se.TarNet, truth = 10),
            Bias = calculate_bias(TarNet, 10),
            RMSE = calculate_mse(TarNet, 10)) %>% 
  ungroup() %>% 
  pivot_longer(!c(N, type)) %>%
  mutate(
    name = factor(name, levels = c("Bias", "RMSE", "Coverage"))
  ) %>%
  ggplot(aes(x = N, y = value, color = type)) + 
  geom_point(aes(shape = type)) +
  geom_line() + 
  facet_wrap(~name, scales = "free_y") + 
  theme_classic() +
  scale_y_continuous(limits = function(x) {
    if(all(x == 0)) {
      return(c(0, 1))
    } else if (min(x) >= 0 & max(x) <= 1) {
      return(c(0, 1))
    } else {
      return(x)
    }
  }) +
  xlab("Sample Size") +
  ylab("") +
  geom_hline(data = data.frame(name = factor("Coverage", levels = c("Bias", "RMSE", "Coverage")), 
                               value = c(0.95)),
             aes(yintercept = value), 
             linetype = "dotted", 
             color = "black") +
  scale_linetype_manual(values = c("solid", "dashed")) +
  theme(legend.position = "bottom") + 
  scale_shape_manual(values = c(16, 12)) +
  theme(legend.title = element_blank()) +
  scale_color_manual(values = c("GPI (new texts)" = "red", 
                                "GPI (text reuse)" = "yellow4"))
