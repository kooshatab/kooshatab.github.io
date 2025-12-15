#R Shiny App:
#https://kooshatab.shinyapps.io/ABALONE/

#Hugging Face RAG App:
#https://huggingface.co/spaces/Kooshatab/abalone-rag-assistant
---
title: "EDA"
author: "Koosha"
date: "2025-12-08"
output:
  pdf_document:
    latex_engine: xelatex
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
```{r}
if (!tinytex::is_tinytex()) tinytex::install_tinytex()
if (!require(caret)) install.packages("caret"); library(caret)
if (!require(Metrics)) install.packages("Metrics"); library(Metrics)
library(tidyverse)
library(glmnet)   # Elastic Net engine

```

```{r}
library(tidyverse)

# Load data
file_path <- file.choose()
df <- read.csv(file_path)

# Remove id if present
if ("id" %in% names(df)) df <- df %>% select(-id)

df$Sex <- as.factor(df$Sex)
```

```{r}
# 📊 **CHUNK 2 — Boxplots for ALL numeric features**

#This creates one boxplot per feature.

# Select numeric features
numeric_vars <- df %>% select(where(is.numeric))

# Long format for ggplot
df_long <- numeric_vars %>% 
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")

# Boxplots for all numeric variables
ggplot(df_long, aes(x = Feature, y = Value)) +
  geom_boxplot(fill = "#A50021", alpha = 0.7, color = "black") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of All Numeric Features", x = "Feature", y = "Value")
```

```{r}
# 📦 **CHUNK 3 — Boxplots Grouped by Sex (Multivariate EDA)**

#Shows how each feature varies by Sex.


# Pivot numeric features but keep Sex
df_long_sex <- df %>%
  select(Sex, where(is.numeric)) %>%
  pivot_longer(cols = -Sex, names_to = "Feature", values_to = "Value")

ggplot(df_long_sex, aes(x = Sex, y = Value, fill = Sex)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ Feature, scales = "free_y") +
  scale_fill_manual(values = c("#A50021", "#999999", "#555555")) +
  theme_bw() +
  labs(title = "Boxplots of Numeric Features Grouped by Sex", x = "Sex", y = "Value")
```

```{r}
library(dplyr)
library(ggplot2)
library(scales)

eps <- 1e-6

# ---- Add log-transformed weight features ----
df2 <- df %>%
  mutate(
    log_Weight         = log(Weight + eps),
    log_Shucked_Weight = log(Shucked.Weight + eps),
    log_Viscera_Weight = log(Viscera.Weight + eps),
    log_Shell_Weight   = log(Shell.Weight + eps)
  )

# ---- Correlation with Age (numeric only) ----
num_df <- df2 %>% select(where(is.numeric))
cor_age <- cor(num_df, use = "pairwise.complete.obs")[, "Age"]
cor_age <- cor_age[names(cor_age) != "Age"]

cor_df <- data.frame(
  Feature = names(cor_age),
  Correlation = as.numeric(cor_age)
) %>%
  mutate(absCorr = abs(Correlation)) %>%
  arrange(desc(absCorr))

# ---- Heatmap-colored horizontal bar chart ----
ggplot(cor_df, aes(x = absCorr, y = reorder(Feature, absCorr), fill = Correlation)) +
  geom_col(width = 0.75) +
  scale_fill_viridis_c(option = "plasma") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey40")
  ) +
  labs(
    title = "Key Determinants of Abalone Age",
    subtitle = "Features ranked by absolute correlation with age",
    x = "Correlation with Age",
    y = "",
    fill = "Correlation"
  )

```
```{r}
library(dplyr)
library(ggplot2)
library(tidyr)

eps <- 1e-6

# Prepare data
plot_df <- df %>%
  mutate(
    log_Weight = log(Weight + eps)
  ) %>%
  select(Age, Sex, Weight, log_Weight) %>%
  pivot_longer(
    cols = c(Weight, log_Weight),
    names_to = "Feature",
    values_to = "Value"
  ) %>%
  mutate(
    Feature = recode(
      Feature,
      Weight = "Raw Weight",
      log_Weight = "Log(Weight)"
    )
  )

# Plot
ggplot(plot_df, aes(x = Age, y = Value, color = Sex)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ Feature, scales = "free_y") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Effect of Log Transformation on Weight–Age Relationship",
    subtitle = "Left: Raw Weight vs Age | Right: Log(Weight) vs Age (colored by Sex)",
    x = "Age",
    y = "Weight",
    color = "Sex"
  )

```



```{r}
# ------------------------------------------------------------
# Bar chart of correlations with Age (sorted)
# ------------------------------------------------------------

# Compute correlations of all numeric predictors with Age
numeric_vars <- df %>% dplyr::select(where(is.numeric))
cor_age <- cor(numeric_vars, use = "pairwise.complete.obs")["Age", ]

# Convert to data frame and remove Age itself
cor_df <- data.frame(
  Feature = names(cor_age),
  Correlation = as.numeric(cor_age)
) %>%
  dplyr::filter(Feature != "Age") %>%
  dplyr::arrange(desc(abs(Correlation)))

# Coolwarm color scale mapped to correlation values
coolwarm_palette <- colorRampPalette(c("#3B4CC0", "white", "#B40426"))(200)

ggplot(cor_df, aes(x = reorder(Feature, Correlation), y = Correlation, fill = Correlation)) +
  geom_col(color = "black", width = 0.7) +
  coord_flip() +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major = element_line(color = "grey85", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "none"
  ) +
  labs(
    title = "Correlation of Numeric Features with Age (Sorted)",
    y = "Correlation"
  )
```


```{r}
# 🔗 **CHUNK 5 — Pairwise Scatterplot Matrix**
# ------------------------------------------------------------
# Scatterplots: Age vs each numeric feature (faceted)
# ------------------------------------------------------------
# ------------------------------------------------------------
# Scatterplots: Age vs each numeric feature (faceted by feature)
# ------------------------------------------------------------

df_long_scatter <- df %>%
  select(Sex, Age, where(is.numeric)) %>%
  pivot_longer(
    cols = -c(Sex, Age),
    names_to = "Feature",
    values_to = "Value"
  )

ggplot(df_long_scatter, aes(x = Value, y = Age, color = Sex)) +
  geom_point(alpha = 0.5) +
  # Explicit formula so ggplot doesn't print the info message
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.7) +
  facet_wrap(~ Feature, scales = "free_x") +
  theme_gray() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  ) +
  labs(
    title = "Age vs Numeric Features by Sex",
    x = "Feature Value",
    y = "Age"
  )


```
```{r}
# ------------------------------------------------------------
# Load dataset via file chooser
# ------------------------------------------------------------
file_path <- file.choose()
df <- read.csv(file_path)

df <- df %>% select(-id)
# ------------------------------------------------------------
# Keep ONLY selected features
# ------------------------------------------------------------
df$Sex <- as.factor(df$Sex)
df$Sex <- relevel(df$Sex, ref = "I")

# ------------------------------------------------------------
# Create model matrix (dummy encoding for Sex)
# ------------------------------------------------------------
X <- model.matrix(Age ~ ., df)[, -1]
y <- df$Age

data_ml <- data.frame(Age = y, X)

# ------------------------------------------------------------
# 5-fold CV repeated 5 times
# ------------------------------------------------------------
set.seed(42)
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5,
  summaryFunction = defaultSummary
)

mlr_repeatedcv_model <- train(
  Age ~ ., 
  data = data_ml,
  method = "lm",
  trControl = train_control,
  metric = "MAE"
)

# ------------------------------------------------------------
# Show results
# ------------------------------------------------------------
mlr_repeatedcv_model
```



```{r}
# ------------------------------------------------------------
# Load dataset via file chooser
# + log-transform weight features
# ------------------------------------------------------------
file_path <- file.choose()
df <- read.csv(file_path)

# Drop id
df <- df %>% select(-id)

# Make column names R-safe (handles spaces like "Shell Weight")
names(df) <- make.names(names(df))

# Sex as factor
df$Sex <- as.factor(df$Sex)

```

```{r}
# ------------------------------------------------------------
# Log-transform weight features (ONLY change requested)
# ------------------------------------------------------------
eps <- 1e-6

df <- df %>%
  mutate(
    # --- Log-transformed weight features ---
    log_Weight         = log(Weight + eps),
    log_Shucked_Weight = log(Shucked.Weight + eps),
    log_Viscera_Weight = log(Viscera.Weight + eps),
    log_Shell_Weight   = log(Shell.Weight + eps),

    # --- Ellipsoid volume approximation ---
    Volume = (pi / 6) * Length * Diameter * Height,

    # --- Density (biomass per unit volume) ---
    Density = Weight / (Volume + eps),

    # (optional but recommended)
    log_Density = log(Density + eps)
  )

```

```{r}
# ------------------------------------------------------------
# Create model matrix (dummy encoding for Sex)
# ------------------------------------------------------------
X <- model.matrix(Age ~ ., df)[, -1]
y <- df$Age

data_ml <- data.frame(Age = y, X)
```


```{r}
# ------------------------------------------------------------
# 5-fold CV repeated 5 times for Elastic Net
# ------------------------------------------------------------
# Custom summary with MAE, RMSE, R2
mae_summary <- function(data, lev = NULL, model = NULL) {
  out <- c(
    MAE = Metrics::mae(data$obs, data$pred),
    RMSE = Metrics::rmse(data$obs, data$pred),
    Rsquared = cor(data$obs, data$pred)^2
  )
  return(out)
}

set.seed(42)
train_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  summaryFunction = mae_summary
)

elastic_model <- train(
  Age ~ .,
  data = data_ml,
  method = "glmnet",
  preProcess = c("center", "scale"),   # ✅ scaling happens AFTER feature engineering
  trControl = train_control,
  metric = "MAE",
  tuneLength = 10
)

```

```{r}
# ------------------------------------------------------------
# Show results
# ------------------------------------------------------------
# Show the tuning results
elastic_model$results

# Extract the minimum MAE across all alpha/lambda combinations
min_mae <- min(elastic_model$results$MAE, na.rm = TRUE)
min_mae

```

```{r}
# ------------------------------------------------------------
# Scatterplots: log(weight features) vs Age, by Sex
# ------------------------------------------------------------
library(ggplot2)
library(tidyr)
library(dplyr)

# Select variables for plotting
plot_df <- df %>%
  select(
    Age,
    Sex,
    log_Weight,
    log_Shucked_Weight,
    log_Viscera_Weight,
    log_Shell_Weight
  ) %>%
  pivot_longer(
    cols = starts_with("log_"),
    names_to = "Feature",
    values_to = "Value"
  )

# Scatterplots with linear trend
ggplot(plot_df, aes(x = Value, y = Age, color = Sex)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Feature, scales = "free_x") +
  labs(
    title = "Log-Transformed Weight Features vs Age by Sex",
    x = "Log-transformed weight",
    y = "Age"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

```












