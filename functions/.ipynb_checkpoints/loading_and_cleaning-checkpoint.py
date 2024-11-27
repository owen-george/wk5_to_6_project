import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

client_data = "../data/raw/df_final_demo.csv"
experiment_data = "../data/raw/df_final_experiment_clients.csv"
web_data1 = "../data/raw/df_final_web_data_pt_1.csv"
web_data2 = "../data/raw/df_final_web_data_pt_2.csv"

client_df = pd.read_csv(client_data)
experiment_df = pd.read_csv(experiment_data)
web1_df = pd.read_csv(web_data1)
web2_df = pd.read_csv(web_data2)

def merge_and_clean_client_df(client_df = client_df, experiment_df = experiment_df):
    ''' Merges client and experiment dataframes and adds balance quartiles ''' 
    
    import pandas as pd
    merged_client_df = pd.merge(client_df, experiment_df, on='client_id', how='outer')
    client_df_cleaned = merged_client_df[merged_client_df.isnull().sum(axis=1) <= 5]
    client_df_cleaned['balance_quartile'] = pd.qcut(client_df_cleaned['bal'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return client_df_cleaned

def merge_and_clean_web_df(web1_df = web1_df, web2_df = web2_df, experiment_df = experiment_df):
    '''Combines web_data together and merges with experiment data
    Gets date_time into appropriate format
    Sorts dataframe by time within each visit
    Drops web data without distinction between Control/Test''' 
    import pandas as pd
    web_df = pd.concat([web1_df, web2_df], axis=0, ignore_index=True)
    web_exp_df = pd.merge(web_df, experiment_df, on='client_id', how='right')
    web_exp_df['date_time'] = pd.to_datetime(web_exp_df['date_time'], format='%d/%m/%y %H:%M:%S')
    web_exp_df = web_exp_df.sort_values(by=['client_id', 'visit_id', 'date_time'])
    web_exp_df = web_exp_df.dropna(subset=['Variation'])
    return (web_exp_df)

def remove_duplicates(df):
    ''' Removes duplicated "start" and "confirm" steps, only keeping the last in each visit ''' 
    new_df = df.copy()
    
    non_start_df = new_df[new_df['process_step'] != 'start']
    filt_start = new_df[new_df['process_step'] == 'start'].drop_duplicates(subset=['visit_id'], keep='last')
    new_df = pd.concat([non_start_df, filt_start])
    new_df = new_df.sort_values(by=['client_id', 'visit_id', 'date_time'])
    
    non_confirm_df = new_df[new_df['process_step'] != 'confirm']
    filt_confirm = new_df[new_df['process_step'] == 'confirm'].drop_duplicates(subset=['visit_id'], keep='last')

    new_df = pd.concat([non_confirm_df, filt_confirm])
    new_df = new_df.sort_values(by=['client_id', 'visit_id', 'date_time'])
    return(new_df)

def step_checks(df):
    ''' Adds columns to dataframe detailing if the step successfully proceeded to the next step,
    if the step was the final one in the visit, or if the step was an error (went back a stage)'''
    
    new_df = df.copy()
    new_df
    new_df = new_df.sort_values(by=['client_id', 'visit_id', 'date_time'])

    #Adds a column to say if that step is the final step of the visit
    new_df['visit_final_step'] = new_df['visit_id'] != new_df['visit_id'].shift(-1)

    #Adds a column for the duration (in seconds) of each step
    new_df['step_duration'] = (new_df['date_time'].shift(-1) - new_df['date_time']).dt.total_seconds()
    new_df.loc[new_df['visit_final_step'] == True, 'step_duration'] = np.nan

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']
    next_step = {'start': 'step_1', 'step_1': 'step_2', 'step_2': 'step_3', 'step_3': 'confirm'}

    #Adds "next_process_step" column with the step from the following row
    new_df['next_process_step'] = new_df['process_step'].shift(-1)
    
    #Adds "step_proceeds" column which checks if the next step is the same visit and moved forward a step
    new_df['step_proceeds'] = new_df.apply(
        lambda row: row['next_process_step'] == next_step.get(row['process_step'], None),
        axis=1
    )
    new_df.loc[new_df['visit_final_step'] == True, 'step_proceeds'] = False

    #Adds "step_error" column which checks if the next step is the same visit and moved back a step
    new_df['step_error'] = new_df.apply(
        lambda row: row['process_step'] == next_step.get(row['next_process_step'], None),
        axis=1
    )
    new_df.loc[new_df['visit_final_step'] == True, 'step_error'] = False

    #Removes "next_process_step" column with the step from the following row
    new_df.drop(columns='next_process_step', inplace=True)
    new_df.reset_index(inplace=True)
    new_df.drop(columns='index', inplace=True)
    return new_df

def remove_outlier_visits(df):
    ''' Within each Variation and step removes visits containing steps of outlier durations
    outliers defined as being 1.5 * IQR below Q1 or above Q3'''
    def duration_outliers(group):
        Q1 = group['step_duration'].quantile(0.25)
        Q3 = group['step_duration'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter rows where step_duration is an outlier
        outliers = group[(group['step_duration'] < lower_bound) | (group['step_duration'] > upper_bound)]
        return outliers

    outliers_df = df.groupby(['Variation', 'process_step'], group_keys=False).apply(duration_outliers)

    outlier_visits = list(outliers_df['visit_id'].unique())

    filt_df = df[~df['visit_id'].isin(outlier_visits)]
    
    return filt_df

def successful_visit_col(df=analysis_df):
    ''' Adds a column to the dataframe stating if the visit was a success (Processed every step from start to confirm) '''
    analysis_df = df.copy()
    start_visit_ids = list(analysis_df[(analysis_df['process_step'] == 'start') & (analysis_df['step_proceeds'] == True)]['visit_id'])
    step_1_visit_ids = list(analysis_df[(analysis_df['process_step'] == 'step_1') & (analysis_df['step_proceeds'] == True)]['visit_id'])
    step_2_visit_ids = list(analysis_df[(analysis_df['process_step'] == 'step_2') & (analysis_df['step_proceeds'] == True)]['visit_id'])
    step_3_visit_ids = list(analysis_df[(analysis_df['process_step'] == 'step_3') & (analysis_df['step_proceeds'] == True)]['visit_id'])
    confirm_visit_ids = list(analysis_df[analysis_df['process_step']=='confirm']['visit_id'])
    successful_visits = list(set(start_visit_ids) & set(step_1_visit_ids) & set(step_2_visit_ids) & set(step_3_visit_ids) & set(confirm_visit_ids))
    analysis_df['successful_visit'] = analysis_df['visit_id'].isin(successful_visits)
    return analysis_df

# Runs all the cleaning and loading code and saves the csvs
client_df_cleaned = merge_and_clean_client_df()
client_df_cleaned.to_csv('../data/clean/clean_client_data.csv', index=False)

web_exp_df = merge_and_clean_web_df()
clean_df = remove_duplicates(web_exp_df)
analysis_df = step_checks(clean_df)

no_outlier_df = remove_outlier_visits(analysis_df)
no_outlier_df.to_csv('../data/clean/removed_outliers_data.csv', index=False)

extra_visit_col = successful_visit_col(no_outlier_df)
extra_visit_col.to_csv('../data/clean/data_extra_col.csv', index=False)