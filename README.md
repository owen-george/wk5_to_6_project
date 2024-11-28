# wk5_to_6_project

This project is an A/B testing analysis aimed at comparing the performance of a new UI and process vs Control groups based on user behavior, demographics, and engagement metrics. The analysis evaluates whether key metrics differ significantly between the two groups using statistical tests and visualization techniques.

## Project Overview

The primary objectives of this project are:
- **Understand user behavior**: Analyse user engagement and success rates in Test and Control groups.
- **Evaluate demographic patterns**: Investigate differences in user demographics (e.g., age, gender, and client tenure) across variations.
- **Statistical significance testing**: Determine whether observed differences are statistically significant using chi-square and t-tests.
- **Visualize data distributions**: Present key insights through KDE plots and summary statistics.

### Key Questions Addressed
1. Is there a significant improvement in success rates (>5%) between Test and Control groups?
2. Is the experiment appropriate to make a conclusion?
3. Was the new process quicker than the old process?

## Data Description

The dataset consists of three primary files:
- **Client Data**: Demographic information (e.g., age, tenure, balance).
- **Visit Data**: Details of user visits, including success rates and engagement metrics.
- **Experiment Data**: Mapping between client IDs and test/control group assignments.

## Analysis Workflow

1. **Loading and Cleaning Data**
   - Merge the client, visit, and analysis datasets.
   - Handle duplicate columns and rename for clarity.
   - Remove missing values and outliers where necessary for statistical tests.

2. **Exploratory Data Analysis (EDA)**
   - Summarized distributions of key variables like balance and tenure.
   - Identified potential outliers using the IQR method.

3. **Statistical Tests**
   - **Chi-square tests**: Assessed independence between categorical variables like gender and success rates across variations.
   - **T-tests**: Compared numerical metrics (e.g. step count, balance, visit duration, etc.) between Test and Control groups.

4. **Visualization**
   - Generated KDE plots to compare distributions of numerical variables such as balance, age, and tenure.
   - Plotted filtered distributions after removing outliers for better interpretability.

## Key Findings

- **Test and Control grouos**: No significant difference between Test and Control groups based on chi-square results.
- **Success Rates**: Significant association between Variation (Test/Control) and successful visits.
- **Metric Comparisons**: Some metrics, like `step_count` and `visit_duration`, showed statistically significant differences, while others did not.

## Statistical Methods Used

### Chi-square Test
- Hypothesis:
  - **H0**: There is no association between the two categorical variables.
  - **H1**: There is an association between the two categorical variables.

### Independent T-Test
- Hypothesis:
  - **H0**: Mean values of the metric are the same for Test and Control groups.
  - **H1**: Mean values of the metric differ between Test and Control groups.
- Assumes unequal variances between groups (`equal_var=False`).

### Outlier Removal
- Applied the IQR method to filter outliers for numerical columns, ensuring more robust statistical analysis.

## Visualization Samples

KDE plots were generated for key variables to visualize differences between Test and Control groups. Examples include:

- **Age Distribution**:
  ![Age Distribution](figures/age_dist_control_vs_test.png)
  
- **Balance Distribution (Filtered)**:
  ![Filtered Balance Distribution](figures/bal_dist_control_vs_test_filtered.png)
  
- **Client Tenure Distribution**:
  ![Client Tenure Distribution](figures/clnt_tenure_dist_control_vs_test.png)

- **Visit Duration Distribution**:
 ![Visit Duration Distribution](figures/visit_duration_control_vs_test.png)

## Requirements

To run the analysis, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scipy`

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
