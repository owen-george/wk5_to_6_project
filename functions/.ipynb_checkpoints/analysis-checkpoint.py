import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import chi2_contingency

#File paths
client_data = "../data/clean/clean_client_data.csv"
analysis_data = "../data/clean/data_extra_col.csv"

# Load datasets
def load_data(client_data_path=client_data, analysis_data_path=analysis_data):
    """
    Load the client and analysis data from CSV files.

    Returns:
    - client_df (DataFrame): Client data.
    - analysis_df (DataFrame): Analysis data.
    """
    client_df = pd.read_csv(client_data_path)
    analysis_df = pd.read_csv(analysis_data_path)
    return client_df, analysis_df

client_df, analysis_df = load_data()

# Create pivot table for step remain analysis
def step_remain_analysis(df):
    """
    Generate a pivot table for step remain analysis.
    """
    step_remain_pivot = df.pivot_table(
        index='process_step',
        columns='Variation',
        values=['visit_final_step'],
        aggfunc=['mean', 'count']
    ).round(4)
    step_remain_pivot.columns = step_remain_pivot.columns.droplevel(1)
    return step_remain_pivot

step_remain_pivot = step_remain_analysis(analysis_df)

# Calculate success metrics
def calculate_success_metrics(step_remain_pivot):
    """
    Calculate success metrics for control and test groups.
    """
    control_confirm = step_remain_pivot[('count', 'Control')].loc['confirm']
    control_start = step_remain_pivot[('count', 'Control')].loc['start']
    test_confirm = step_remain_pivot[('count', 'Test')].loc['confirm']
    test_start = step_remain_pivot[('count', 'Test')].loc['start']

    control_success = control_confirm / control_start
    test_success = test_confirm / test_start

    success_improvement = (test_success / control_success - 1).round(3)

    print(f"Control group visits {100 * control_success}%")
    print(f"Success rate improved by {100 * success_improvement}%")

    return success_improvement

calculate_success_metrics(step_remain_pivot)


def successful_visit_durations(df=analysis_df):
    """
    Analyze and compute average successful visit durations.
    """
    
    experiment_df = df.copy()
    success_df = experiment_df[experiment_df['successful_visit']==True]
    duration_df = success_df[(success_df['process_step']=='start') | (success_df['process_step']=='confirm')]
    duration_df = duration_df[['visit_id', 'Variation', 'process_step', 'date_time']]

    duration_df_pivot = duration_df.pivot_table(index=['Variation', 'visit_id'],
                                            columns='process_step',
                                            values='date_time',
                                            aggfunc='first')

    duration_df_pivot['confirm'] = pd.to_datetime(duration_df_pivot['confirm'])
    duration_df_pivot['start'] = pd.to_datetime(duration_df_pivot['start'])
    duration_df_pivot['duration'] = (duration_df_pivot['confirm'] - duration_df_pivot['start']).dt.total_seconds()

    success_durations = duration_df_pivot.reset_index()[['Variation', 'visit_id', 'duration']]

    results = success_durations.pivot_table(index=['Variation'],
                                            values='duration',
                                            aggfunc='mean').round(2)

    result_dict = results.to_dict()['duration']
    
    return(result_dict)

    
def prepare_visit_df(df=analysis_df):

    piv_df = df.pivot_table(
        index=['visit_id', 'client_id', 'Variation','successful_visit'],
        values=['step_duration', 'process_step'],
        aggfunc={
            'step_duration': 'sum',
            'process_step': 'count',
        }).round(2)
    
    results = piv_df.reset_index()
    results = results.rename(columns={'process_step': 'step_count', 'step_duration': 'visit_duration'})

    piv_df2 = analysis_df.pivot_table(
        index='visit_id', 
        columns='process_step', 
        aggfunc='size', 
        fill_value=0)

    merged_piv_df = pd.merge(results, piv_df2, on='visit_id', how='outer')
    
    return(merged_piv_df)

visit_df = visit_df(analysis_df)

visit_df.to_csv('../data/clean/visit_data.csv', index=False)

def analyze_variation_success(visit_df):
    """
    Generate summary metrics for confirmed and fully successful visits.
    """
    visit_piv = visit_df.pivot_table(
        index='Variation',
        values=['visit_id', 'confirm', 'successful_visit'],
        aggfunc={
            'visit_id': 'count',
            'confirm': 'sum',
            'successful_visit': 'sum'
        }
    )

    visit_piv = visit_piv.rename(
        columns={
            'visit_id': 'visit_count',
            'confirm': 'confirmed_visits',
            'successful_visit': 'full_success_visits'
        }
    )

    visit_piv['full_success_visits'] = (100 * visit_piv['full_success_visits'] / visit_piv['visit_count']).round(2)
    visit_piv['confirmed_visits'] = (100 * visit_piv['confirmed_visits'] / visit_piv['visit_count']).round(2)

    return visit_piv[['confirmed_visits', 'full_success_visits']]

analyze_variation_success(visit_df)

# Visualize data
def plot_boxplot(visit_df, output_path='../figures/boxplot.png'):
    """
    Create and save a boxplot for visit durations by variation.
    """
    sns.boxplot(
        data=visit_df[visit_df['successful_visit'] == True],
        y='visit_duration',
        x='Variation',
        hue='Variation',
        palette="coolwarm",
        legend=False
    )
    plt.savefig(output_path)

# Pivot table for process step analysis
def pivot_process_step(df=analysis_df):
    """
    Generate a pivot table summarizing process step metrics by variation.
    """
    
    df['other_issue'] = (df['step_proceeds']==False) & (df['step_error']==False) & (df['visit_final_step']==False)
    
    piv_df = df.pivot_table(index='process_step',
                columns='Variation',
                values=['step_duration', 'step_proceeds', 'visit_final_step', 'step_error', 'other_issue', 'visit_id'],
                aggfunc={
                    'step_duration': 'mean',
                    'step_proceeds': 'mean',
                    'visit_final_step': 'mean',
                    'step_error': 'mean',
                    'other_issue': 'mean',
                    'visit_id' : 'nunique'
                }).round(2)

    piv_df = piv_df.sort_values(by=('visit_id', 'Test'), ascending=False)    

    return piv_df


    
    piv_df = piv_df[['step_duration', 'step_proceeds', 'visit_final_step', 'step_error', 'other_issue', 'visit_id' ]].sort_values(by=('visit_id', 'Test'), ascending=False)
    
    return(piv_df)

# Visit success rate by variation
def visit_success_rate(df=analysis_df):
    """
    Calculate success rates for each variation.
    """
    piv_df = df.pivot_table(
        index='Variation',
        columns = 'successful_visit',
        values='visit_id',
        aggfunc='nunique')

    results = piv_df.reset_index()
    results = results.rename(columns={'visit_id': 'visit_count'})
    results['success_rate'] = round(100 * results[True] / (results[True] + results[False]),2)
    results.set_index('Variation', inplace=True)
   
    result_dict = results.to_dict()['success_rate']

    return(result_dict)
    
# Improvement metrics
def calculate_improvements(df):
    """
    Calculate percentage improvements for success rate and duration.
    """
    duration_imp = round(100 * (successful_visit_durations(df)['Test'] / successful_visit_durations(df)['Control'] - 1), 2)
    success_imp = round(100 * (calculate_visit_success_rate(df)['Test'] / calculate_visit_success_rate(df)['Control'] - 1), 2)

    print(f"Visit success rate changed: {success_imp}%")
    print(f"Visit average duration changed: {duration_imp}%")
    return duration_imp, success_imp

# Violin plots for step durations
def plot_violin(df, output_path):
    """
    Create violin plots for step durations by variation.
    """
    sns.violinplot(
        data=df[(df['Variation'] == 'Control')],
        y='step_duration',
        x='process_step',
        palette="coolwarm",
        hue='process_step',
        legend=False
    )
    plt.savefig(f"{output_path}_control.png", transparent=True)
    plt.clf()

    sns.violinplot(
        data=df[df['Variation'] == 'Test'],
        y='step_duration',
        x='process_step',
        palette="coolwarm",
        hue='process_step',
        legend=False
    )
    plt.savefig(f"{output_path}_test.png", transparent=True)

error_visit_ids = list(analysis_df[(analysis_df['step_error'] == True)]['visit_id'].unique())
non_proceed_visit_ids = list(analysis_df[(analysis_df['step_proceeds'] == False) & (analysis_df['visit_final_step'] == False)]['visit_id'].unique())
total_visit_ids = list(analysis_df['visit_id'].unique())

import numpy as np

# Assuming 'results' is the dataframe with the given data
test_visits = visit_df[visit_df['Variation'] == 'Test']
control_visits = visit_df[visit_df['Variation'] == 'Control']

# Calculate the number of successes (True values for 'successful_visit')
test_success = test_visits[test_visits['successful_visit'] == True].shape[0]
test_total = test_visits.shape[0]

control_success = control_visits[control_visits['successful_visit'] == True].shape[0]
control_total = control_visits.shape[0]

# Calculate the completion rates (proportions)
test_success_rate = test_success / test_total
control_success_rate = control_success / control_total

print(f"Test Completion Rate: {test_success_rate:.4f}")
print(f"Control Completion Rate: {control_success_rate:.4f}")

test_samples = [np.mean(np.random.choice(test_visits['successful_visit'], 100))for _ in range(500)]
control_samples = [np.mean(np.random.choice(control_visits['successful_visit'], 100))for _ in range(500)]


st.ttest_ind(test_samples, control_samples, equal_var=False, alternative="two-sided")

success_df = visit_df[visit_df['successful_visit']==True]

test_success_durations = success_df[success_df['Variation'] == 'Test']['visit_duration']
control_success_durations = success_df[success_df['Variation'] == 'Control']['visit_duration']

# KDE plot for durations
def plot_kde(control_durations, test_durations, output_path):
    """
    Create and save a KDE plot for visit durations.

    Parameters:
    - control_durations (array-like): Visit durations for the control group.
    - test_durations (array-like): Visit durations for the test group.
    - output_path (str): File path to save the plot.
    """
    sns.kdeplot(control_durations, label='Control', fill=True)
    sns.kdeplot(test_durations, label='Test', fill=True)
    plt.xlabel('Duration')
    plt.ylabel('Density')
    plt.title('KDE Plot for Control vs Test Successful Visit Duration')
    plt.legend()
    plt.savefig(output_path, transparent=True)
    plt.clf()


# Merge DataFrames for analysis
def merge_client_data(analysis_df, client_df, visit_df):
    """
    Merge analysis, client, and visit data for comprehensive analysis.
    """
    analysis_client = pd.merge(analysis_df, client_df, on='client_id', how='left')
    analysis_client.drop(columns=['Variation_y'], inplace=True)
    analysis_client.rename(columns={'Variation_x': 'Variation'}, inplace=True)

    visit_id_df = analysis_df[['visit_id', 'client_id']].drop_duplicates()
    visit_client = pd.merge(visit_df, visit_id_df, on=['visit_id', 'client_id'], how='left')
    visit_client = pd.merge(visit_client, client_df, on='client_id', how='left')
    visit_client.drop(columns=['Variation_y'], inplace=True)
    visit_client.rename(columns={'Variation_x': 'Variation'}, inplace=True)

    return visit_client

# Is gendr similar between Test and Control
# Chi-square test for categorical variables
def perform_chi2_test(df, col1, col2):
    """
    Perform a chi-square test for independence between two categorical variables.

    Parameters:
    - df (DataFrame): The input data.
    - col1 (str): First categorical column.
    - col2 (str): Second categorical column.

    Returns:
    - chi2_statistic (float): Chi-square statistic.
    - chi2_p_value (float): P-value from the test.
    """
    crosstab_res = pd.crosstab(df[col1], df[col2])
    chi2_statistic, chi2_p_value, _, _ = chi2_contingency(crosstab_res)
    print(crosstab_res)
    print(f"Chi-square statistic: {chi2_statistic:.4f}, P-value: {chi2_p_value:.4f}")
    return chi2_statistic, chi2_p_value

# T-test for numerical variables
def perform_ttest(df_test, df_control, alpha=0.05):
    """
    Perform independent t-tests for multiple numerical variables.

    Parameters:
    - df_test (DataFrame): Test group data.
    - df_control (DataFrame): Control group data.
    - alpha (float): Significance level.

    Prints the results for each variable.
    """
    for col in df_test.columns:
        _, p_val = st.ttest_ind(df_test[col], df_control[col], equal_var=False, alternative='two-sided')
        if p_val > alpha:
            print(f"For {col}, with p-value of {p_val:.4f}, we do not reject the null hypothesis.")
        else:
            print(f"For {col}, with p-value of {p_val:.4f}, we reject the null hypothesis.")

# KDE plot for variable distribution
def plot_kde(df, col, output_path):
    """
    Create a KDE plot for a numerical variable grouped by variations.

    Parameters:
    - df (DataFrame): Input data.
    - col (str): Column for plotting.
    - output_path (str): File path to save the plot.
    """
    sns.kdeplot(df[df['Variation'] == 'Test'][col], label='Test', fill=True)
    sns.kdeplot(df[df['Variation'] == 'Control'][col], label='Control', fill=True)
    plt.xlabel(col.capitalize().replace('_', ' '))
    plt.ylabel('Density')
    plt.title(f'{col.capitalize()} Distribution')
    plt.legend()
    plt.savefig(output_path, transparent=True)
    plt.clf()

# Remove outliers from a column
def remove_outliers(df, col):
    """
    Remove outliers from a numerical column based on the IQR method.

    Parameters:
    - df (DataFrame): Input data.
    - col (str): Column to filter.

    Returns:
    - filt_df (DataFrame): DataFrame with outliers removed.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Main execution flow
def analyze_demographics_and_metrics(analysis_df, client_df, visit_df):
    """
    Perform the final stage of analysis, including demographic insights, T-tests, and KDE plots.
    """
    # Merge data
    visit_client = merge_client_data(analysis_df, client_df, visit_df)

    # Chi-square tests
    print("\nChi-square test for gender:")
    perform_chi2_test(visit_client, 'Variation', 'gendr')

    print("\nChi-square test for successful visits:")
    perform_chi2_test(visit_client, 'Variation', 'successful_visit')

    # T-tests for numerical variables
    numerical_cols = ['step_count', 'visit_duration', 'clnt_tenure_yr', 'clnt_tenure_mnth',
                      'clnt_age', 'num_accts', 'bal', 'calls_6_mnth', 'logons_6_mnth']
    df_test = visit_client[visit_client['Variation'] == 'Test'][numerical_cols].dropna()
    df_control = visit_client[visit_client['Variation'] == 'Control'][numerical_cols].dropna()

    print("\nT-test results:")
    perform_ttest(df_test, df_control)

    # KDE plots
    plot_kde(visit_client, 'clnt_age', "../figures/age_dist_control_vs_test.png")
    plot_kde(visit_client, 'bal', "../figures/bal_dist_control_vs_test.png")
    plot_kde(visit_client, 'clnt_tenure_mnth', "../figures/clnt_tenure_dist_control_vs_test.png")

    # Outlier removal and KDE plot for balance
    bal_filtered = remove_outliers(visit_client, 'bal')
    plot_kde(bal_filtered, 'bal', "../figures/bal_dist_control_vs_test_filtered.png")
