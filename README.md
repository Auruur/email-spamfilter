# Spambase Classification Project

**Authors**: Alessandro Soccol, Marco Cosseddu, Giovanni Murgia, Pietro Cosseddu, Arturo Rodriguez  
**Institution**: University of Cagliari  
**Course**: Applied Computer Science and Data Analytics 2022/2023

## Abstract

This project focuses on the Spambase dataset, a binary classification problem where emails are labeled as spam (1) or not spam (0). The goal is to analyze data quality and apply various preprocessing and classification techniques to improve prediction accuracy. We compare five classifiers: three implemented using the Scikit-learn library, and two custom classifiers. The project uses tools such as Pandas, Numpy, and Imbalanced-learn for data handling and preprocessing.

## Table of Contents

1. [Introduction](#introduction)
2. [Preprocessing](#preprocessing)
3. [Classifiers](#classifiers)
    1. [Random Forest](#random-forest)
    2. [k-Nearest Neighbour](#k-nearest-neighbour)
    3. [Decision Tree](#decision-tree)
    4. [Custom Naive Bayes Classifier](#custom-naive-bayes)
    5. [Custom Ensemble Classifier](#custom-ensemble)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction

The Spambase dataset, developed by Hewlett-Packard (HP) Labs in 1999, contains 4601 email samples, classified into spam (39.4%) or non-spam (60.6%). Each sample has 58 features, where the first 48 are word frequency percentages, followed by 6 character frequency percentages. Three additional features represent the length of uninterrupted capital letter sequences, and the last feature labels the email as spam (1) or not spam (0).

## Preprocessing

We performed multiple data preprocessing steps, including:
- **Missing Value Handling**: Checked for missing values using Pandas' `.info()` and `.isnull()` functions.
- **Outlier Detection**: Identified outliers using both percentile-based and boxplot-based methods.
- **Feature Correlation**: We used a correlation matrix to detect highly correlated features for potential removal during feature selection.

### Preprocessing Scripts
- **libraries.py**: Contains imports for Pandas, Numpy, and Imbalanced-learn.
- **preprocessing.py**: Performs data scaling (MinMax Scaler, Standard Scaler, Normalization) and feature selection (VarianceThreshold, SelectKBest, SequentialFeatureSelector).

## Classifiers

We tested five classifiers, applying preprocessing and tuning their hyperparameters using GridSearchCV.

### Random Forest

We trained a **Random Forest Classifier** using Scikit-learn. Hyperparameters tuned included the number of trees (`n_estimators`), splitting criteria (`criterion`), and maximum depth (`max_depth`). We used a 10-fold cross-validation for performance evaluation.

### k-Nearest Neighbour

The **kNN Classifier** was trained using distance-based metrics such as Euclidean and Manhattan distances. Hyperparameter tuning involved adjusting the number of neighbors (`n_neighbors`) and distance weighting schemes (`weights`).

### Decision Tree

We applied a **Decision Tree Classifier**, tuning the criteria for node splitting (`criterion`), maximum tree depth (`max_depth`), and the minimum number of samples required for a split (`min_samples_split`).

### Custom Naive Bayes

We implemented a **Custom Naive Bayes Classifier** tailored to handle continuous attributes using Gaussian distributions for posterior probability calculations. The classifier was tested with and without data preprocessing.

### Custom Ensemble

We created a **Custom Ensemble Classifier** based on majority voting. The ensemble combined predictions from the kNN, Decision Tree, and Naive Bayes classifiers. We tested both hard and soft voting schemes and tuned the voting weights.

## Results

### Classifier Performance

The performance of each classifier was evaluated with and without preprocessing techniques like MinMax Scaler, feature selection, and data balancing. Metrics such as accuracy, precision, recall, and F1-score were calculated. Below are some highlights:

| Classifier       | Accuracy (No Preprocessing) | Accuracy (Preprocessing) |
|------------------|-----------------------------|--------------------------|
| Random Forest     | 0.96                        | 0.98                     |
| k-Nearest Neighbour | 0.85                      | 0.92                     |
| Decision Tree     | 0.92                        | 0.96                     |
| Custom Naive Bayes| 0.84                        | 0.86                     |
| Custom Ensemble   | 0.91                        | 0.95                     |

## Conclusion

Our results suggest that preprocessing techniques like feature selection and data scaling can significantly enhance the performance of classifiers, particularly for more complex models like Random Forests and ensemble methods. The Custom Ensemble Classifier consistently delivered strong performance across different evaluation metrics, making it the most effective method for this dataset.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Numpy Documentation](https://numpy.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
