# Spectrum-Based Fault Localization (SBFL) visualization script
#
# This script is immediately runnable:
# 1) It generates synthetic data matching the requested CSV schema.
# 2) It builds Figure 1 (exam-score boxplots) and Figure 2 (inspect-score percentages).
# 3) It uses ggplot2 for all visualizations.

library(ggplot2)

# -----------------------------------------------------------------------------
# 1. Synthetic data generation
# -----------------------------------------------------------------------------

set.seed(123)

algorithm_levels <- c("DStar", "GenProg", "Jaccard", "Ochiai", "Tarantula")
approach_levels <- c("Statement-Based", "Predicate-Based", "Hybrid")
program_ids <- sprintf("P%02d", 1:12)

# Create a compact dummy data set with the exact requested schema.
# Each row represents one program subject evaluated with one SBFL technique
# under one program-element approach.
dummy_data <- expand.grid(
  Program = program_ids,
  Algorithm = algorithm_levels,
  Approach = approach_levels,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

n_rows <- nrow(dummy_data)

# Generate exam scores on [0, 1], with small technique/approach effects so
# the boxplots have visible structure.
algo_effect <- c(
  DStar = 0.18,
  GenProg = 0.30,
  Jaccard = 0.24,
  Ochiai = 0.28,
  Tarantula = 0.20
)
approach_effect <- c(
  "Statement-Based" = 0.10,
  "Predicate-Based" = 0.16,
  Hybrid = 0.22
)

exam_score_raw <- rbeta(n_rows, shape1 = 2.2, shape2 = 4.8)
dummy_data$Exam_Score <- pmin(
  1,
  pmax(
    0,
    0.55 * exam_score_raw +
      0.25 * unname(algo_effect[dummy_data$Algorithm]) +
      0.20 * unname(approach_effect[dummy_data$Approach]) +
      rnorm(n_rows, mean = 0, sd = 0.03)
  )
)

# Generate inspect booleans. Higher inspect thresholds should usually be easier
# to satisfy, so the TRUE probability increases with the inspect level.
base_inspect_probability <- 0.18 +
  0.10 * (dummy_data$Algorithm %in% c("Ochiai", "DStar")) +
  0.08 * (dummy_data$Approach == "Predicate-Based") +
  0.12 * (dummy_data$Approach == "Hybrid")

prob_1 <- pmin(0.95, base_inspect_probability)
prob_3 <- pmin(0.97, base_inspect_probability + 0.12)
prob_5 <- pmin(0.99, base_inspect_probability + 0.22)
prob_10 <- pmin(1.00, base_inspect_probability + 0.35)

dummy_data$Inspect_1 <- runif(n_rows) < prob_1
dummy_data$Inspect_3 <- runif(n_rows) < prob_3
dummy_data$Inspect_5 <- runif(n_rows) < prob_5
dummy_data$Inspect_10 <- runif(n_rows) < prob_10

# Enforce the requested ordering as factors.
dummy_data$Algorithm <- factor(dummy_data$Algorithm, levels = algorithm_levels)
dummy_data$Approach <- factor(dummy_data$Approach, levels = approach_levels)
dummy_data$Program <- factor(dummy_data$Program, levels = program_ids)

# Optional: write the synthetic data to CSV so the script mirrors the expected
# input format and can be adapted easily for real results.
write.csv(dummy_data, "sbfl_results.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# 2. Figure 1: Exam-score boxplots
# -----------------------------------------------------------------------------

figure1 <- ggplot(dummy_data, aes(x = Algorithm, y = Exam_Score, fill = Algorithm)) +
  geom_boxplot(outlier.alpha = 0.6, width = 0.7) +
  facet_wrap(~ Approach, nrow = 1) +
  scale_x_discrete(limits = algorithm_levels) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  labs(
    title = "Figure 1: Exam scores under different program elements and different SBFL techniques",
    x = "SBFL Technique",
    y = "Exam Score"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "none",
    strip.background = element_rect(fill = "grey92", colour = "grey60"),
    strip.text = element_text(face = "bold")
  )

print(figure1)

# -----------------------------------------------------------------------------
# 3. Figure 2: Inspect-score percentage bar charts
# -----------------------------------------------------------------------------

# Convert the wide TRUE/FALSE inspect columns into long form using base R.
inspect_levels <- c("Inspect_1", "Inspect_3", "Inspect_5", "Inspect_10")
inspect_labels <- c(
  Inspect_1 = "inspect@1",
  Inspect_3 = "inspect@3",
  Inspect_5 = "inspect@5",
  Inspect_10 = "inspect@10"
)

inspect_long <- do.call(
  rbind,
  lapply(inspect_levels, function(column_name) {
    data.frame(
      Program = dummy_data$Program,
      Algorithm = dummy_data$Algorithm,
      Approach = dummy_data$Approach,
      Inspect_Level = factor(inspect_labels[[column_name]], levels = unname(inspect_labels)),
      Is_True = as.integer(dummy_data[[column_name]]),
      stringsAsFactors = FALSE
    )
  })
)

# Aggregate to percentage of TRUE values by Algorithm, Approach, and Inspect level.
inspect_summary <- aggregate(
  Is_True ~ Algorithm + Approach + Inspect_Level,
  data = inspect_long,
  FUN = mean
)
inspect_summary$Percentage <- 100 * inspect_summary$Is_True
inspect_summary$Algorithm <- factor(inspect_summary$Algorithm, levels = algorithm_levels)
inspect_summary$Approach <- factor(inspect_summary$Approach, levels = approach_levels)
inspect_summary$Inspect_Level <- factor(inspect_summary$Inspect_Level, levels = unname(inspect_labels))

figure2 <- ggplot(inspect_summary, aes(x = Inspect_Level, y = Percentage, fill = Inspect_Level)) +
  geom_col(width = 0.72, colour = "grey25") +
  facet_grid(Approach ~ Algorithm) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title = "Figure 2: Percentage of test suites per each subject that achieved the inspect scores",
    x = "Inspect Level",
    y = "Percentage of TRUE Values"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position = "none",
    strip.background = element_rect(fill = "grey92", colour = "grey60"),
    strip.text = element_text(face = "bold")
  )

print(figure2)

# -----------------------------------------------------------------------------
# End of script
# -----------------------------------------------------------------------------
