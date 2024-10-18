# Advanced R Code for Protein Structure Prediction Analysis

# Load required libraries
library(tidyverse)
library(bio3d)
library(ggpubr)
library(caret
library(glmnet)
library(randomForest)
library(e1071)
library(plotly)
library(parallel)
library(doParallel)

# Set up parallel processing
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Function to load and preprocess protein data
load_protein_data <- function(prediction_file, true_structure_file) {
  predictions <- read_csv(prediction_file)
  true_structures <- read_csv(true_structure_file)
  
  # Merge predictions with true structures
  merged_data <- inner_join(predictions, true_structures, by = "protein_id")
  
  # Calculate prediction accuracy
  merged_data <- merged_data %>%
    mutate(accuracy = map2_dbl(predicted_structure, true_structure, 
                               ~sum(.x == .y) / length(.x)))
  
  return(merged_data)
}

# Function to perform statistical analysis
statistical_analysis <- function(data) {
  # Perform t-test to compare accuracy across different protein types
  t_test_result <- t.test(accuracy ~ protein_type, data = data)
  
  # Perform ANOVA to assess the impact of various factors on accuracy
  anova_result <- aov(accuracy ~ protein_type + sequence_length + prediction_method, data = data)
  
  # Perform correlation analysis
  correlation_matrix <- cor(data %>% select_if(is.numeric))
  
  return(list(t_test = t_test_result, anova = anova_result, correlation = correlation_matrix))
}

# Function to create advanced visualizations
create_visualizations <- function(data) {
  # Box plot of accuracy by protein type
  box_plot <- ggplot(data, aes(x = protein_type, y = accuracy, fill = protein_type)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = "Prediction Accuracy by Protein Type", x = "Protein Type", y = "Accuracy")
  
  # Scatter plot of accuracy vs sequence length with smoothed conditional means
  scatter_plot <- ggplot(data, aes(x = sequence_length, y = accuracy, color = protein_type)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = TRUE) +
    theme_minimal() +
    labs(title = "Accuracy vs Sequence Length", x = "Sequence Length", y = "Accuracy")
  
  # 3D scatter plot of accuracy, sequence length, and hydrophobicity
  plot_3d <- plot_ly(data, x = ~sequence_length, y = ~hydrophobicity, z = ~accuracy, 
                     color = ~protein_type, type = "scatter3d", mode = "markers")
  
  return(list(box_plot = box_plot, scatter_plot = scatter_plot, plot_3d = plot_3d))
}

# Function to perform machine learning analysis
ml_analysis <- function(data) {
  # Prepare data for machine learning
  ml_data <- data %>%
    select(accuracy, sequence_length, hydrophobicity, charge, protein_type) %>%
    mutate(protein_type = as.factor(protein_type))
  
  # Split data into training and testing sets
  set.seed(42)
  train_index <- createDataPartition(ml_data$accuracy, p = 0.8, list = FALSE)
  train_data <- ml_data[train_index, ]
  test_data <- ml_data[-train_index, ]
  
  # Train multiple models
  models <- list(
    lm = train(accuracy ~ ., data = train_data, method = "lm"),
    rf = train(accuracy ~ ., data = train_data, method = "rf"),
    svm = train(accuracy ~ ., data = train_data, method = "svmRadial")
  )
  
  # Make predictions on test data
  predictions <- lapply(models, predict, newdata = test_data)
  
  # Calculate performance metrics
  performance <- lapply(predictions, function(pred) {
    data.frame(
      RMSE = RMSE(pred, test_data$accuracy),
      R2 = R2(pred, test_data$accuracy)
    )
  })
  
  return(list(models = models, predictions = predictions, performance = performance))
}

# Function to perform advanced sequence analysis
sequence_analysis <- function(sequences) {
  # Calculate various sequence properties
  properties <- lapply(sequences, function(seq) {
    aa_comp <- compute_aac(seq)
    hydrophobicity <- sum(aa_comp * hydrophobicity_scores) / length(seq)
    charge <- sum(aa_comp * charge_scores)
    complexity <- entropy(table(strsplit(seq, "")[[1]]))
    
    data.frame(hydrophobicity = hydrophobicity, charge = charge, complexity = complexity)
  })
  
  properties_df <- do.call(rbind, properties)
  
  # Perform clustering based on sequence properties
  clustering <- hclust(dist(properties_df))
  
  return(list(properties = properties_df, clustering = clustering))
}

# Main analysis pipeline
main_analysis <- function(prediction_file, true_structure_file, sequence_file) {
  # Load and preprocess data
  data <- load_protein_data(prediction_file, true_structure_file)
  
  # Perform statistical analysis
  stats <- statistical_analysis(data)
  
  # Create visualizations
  plots <- create_visualizations(data)
  
  # Perform machine learning analysis
  ml_results <- ml_analysis(data)
  
  # Perform sequence analysis
  sequences <- read_fasta(sequence_file)
  seq_analysis <- sequence_analysis(sequences$seq)
  
  # Combine all results
  results <- list(
    data = data,
    statistics = stats,
    visualizations = plots,
    ml_results = ml_results,
    sequence_analysis = seq_analysis
  )
  
  return(results)
}

# Run the analysis
results <- main_analysis("predictions.csv", "true_structures.csv", "sequences.fasta")

# Generate report
generate_report <- function(results) {
  rmarkdown::render("protein_analysis_report.Rmd", 
                    params = list(results = results),
                    output_file = "protein_analysis_report.html")
}

generate_report(results)

# Clean up
stopCluster(cl)
