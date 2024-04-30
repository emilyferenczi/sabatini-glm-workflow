#utils for data handling

import os
import csv
import yaml
import numpy as np
import pandas as pd

def predictor_labels(df):
    """
    Identify the individual licks that have specific meaning in the tasks: 
    lick 1, lick 2 and lick 3 are "operant licks" on different training days
    licknon1-3 are all the other licks, last lick is the last lick of a trial
    """

    df_source = df.copy()
    srs_lick = df_source.groupby(['SessionName', 'TrialNumber'])['lick'].cumsum()
    srs_lick_count = srs_lick * df_source['lick']
    df_lick_count_dummies = pd.get_dummies(srs_lick_count, dtype=int).drop(0, axis=1)
    df_lick_count_dummies = df_lick_count_dummies[[1,2,3,]]
    df_lick_count_dummies['non1-3'] = df_source['lick'] - df_lick_count_dummies.sum(axis=1)
    df_lick_count_dummies.columns = [f'lick_{original_column_name}' for original_column_name in df_lick_count_dummies.columns]

    """
    Columns lick and lick_1, lick_2, lick_3, lick_non-1-3 should not all be used together
    as predictors because of multicollinearity.
    """
    df_source = pd.concat([df_source, df_lick_count_dummies], axis=1)

    assert np.all(df_source['lick'] == df_source[['lick_1', 'lick_2', 'lick_3', 'lick_non1-3']].sum(axis=1)),'Column lick should equal the sum of all other lick columns.'

    """
    This is the code to define the last lick
    """
    srs_lick_count = srs_lick_count.reset_index()
    srs_lick_count = srs_lick_count.rename(columns={srs_lick_count.columns[-1]: 'lick'})
    max_values_per_trial = srs_lick_count.groupby('TrialNumber').apply(lambda x: x.loc[x['lick'].idxmax()])

    result_df = pd.DataFrame(columns=['TrialNumber', 'lick', 'Timestamp', 'SessionName'])

    result_rows = []

    # Now you can iterate through unique TrialNumbers and append the results to the list
    unique_trial_numbers = max_values_per_trial.index.unique()
    for trial_number in unique_trial_numbers:
        max_row = max_values_per_trial.loc[trial_number]
        max_value_for_trial = max_row['lick']
        max_timestamp_for_trial = max_row['Timestamp']
        session_name_for_trial = max_row['SessionName']
        result_rows.append({'TrialNumber': trial_number, 'sum_last_lick': max_value_for_trial, 'Timestamp': max_timestamp_for_trial, 'SessionName': session_name_for_trial})

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(result_rows)
    
    result_df_reset = result_df.reset_index()

    # Iterate over each row in result_df and update df_source where SessionName and Timestamp match
    for _, row in result_df_reset.iterrows():
        session_name = row['SessionName']
        timestamp = row['Timestamp']
        mask = (df_source.index.get_level_values('SessionName') == session_name) & (df_source.index.get_level_values('Timestamp') == timestamp)
        if mask.any():
            df_source.loc[mask, 'sum_last_lick'] = row['sum_last_lick']
     
    #reset multi-index
    df_source['last_lick'] = (df_source['sum_last_lick'] > 0).astype(int)
    
    #df_source.set_index(['SessionName', 'Timestamp'], inplace=True)
    reIndex = df_source.groupby(level=[0, 1])

    """
    Use this code to calculate the numbers of any specific type of trial by outcome
    """
    #trialType = 'H'
    #filtered_df = df_source[df_source['outcome']==trialType]
    #unique_trial_numbers = pd.unique(filtered_df.index.get_level_values('TrialNumber'))
    #unique_trial_numbers
    #print(len(unique_trial_numbers))
    

    """
    Define cue condition: either go or no go
    """
    conditions_cue = (df_source['go'] == 1) + (df_source['nogo'] == 1)
    df_source['cue'] = conditions_cue.astype(int)
    df_source.loc[(df_source['cue'] == 1)]
    
    
    """
    Define cue outcome conditions: 4 options: go-hit, go-miss, nogo-cr, nogo-fa
    """
    conditions_go_hit = (df_source['go'] == 1) & (reIndex['outcome'].transform('max') == 'H')
    df_source['go_hit'] = conditions_go_hit.astype(int)

    conditions_go_miss = (df_source['go'] == 1) & (reIndex['outcome'].transform('max') == 'M')
    df_source['go_miss'] = conditions_go_miss.astype(int)

    conditions_nogo_cr = (df_source['nogo'] == 1) & (reIndex['outcome'].transform('max') == 'CR')
    df_source['nogo_cr'] = conditions_nogo_cr.astype(int)

    conditions_nogo_fa = (df_source['nogo'] == 1) & (reIndex['outcome'].transform('max') == 'FA')
    df_source['nogo_fa'] = conditions_nogo_fa.astype(int)

    df_source.loc[(df_source['nogo_cr'] == 1)]
    
    
    """
    Define lick1_reward: Hit response with reward delivered at lick 1 (default in early training sessions)
    """
    conditions_lick_1R = (reIndex['outcome'].transform('max') == 'H') & (df_source['lick_1'] == 1)
    df_source['lick_1R'] = conditions_lick_1R.astype(int)
    df_source.loc[(df_source['lick_1R'] == 1)]
    
    
    """
    Define lick3_reward: Hit response with reward delivered at lick 3 (default in advanced training sessions)
    """
    conditions_lick_3R = (reIndex['outcome'].transform('max') == 'H') & (df_source['lick_3'] == 1)
    df_source['lick_3R'] = conditions_lick_3R.astype(int)
    df_source.loc[(df_source['lick_3R'] == 1)]
    
    
    """
    Define catch: absence of reward after a correct hit
    """
    conditions_catch = (reIndex['outcome'].transform('max') == 'C') & (df_source['lick_3'] == 1)
    df_source['lick_3C'] = conditions_catch.astype(int)
    df_source.loc[(df_source['lick_3C'] == 1)]
    
    
    """
    Define non-operant lick 3: lick 3 in any other trial type (not hit trial)
    """
    conditions_lick3_nonOp = (reIndex['outcome'].transform('max') != 'H') & (reIndex['outcome'].transform('max') != 'C') & (df_source['lick_3'] == 1)
    df_source['lick_3NOP'] = conditions_lick3_nonOp.astype(int)
    df_source.loc[(df_source['lick_3NOP'] == 1)]
    
    
    """
    Define miss reward: passive reward given after a miss
    """
    conditions_missReward = (reIndex['outcome'].transform('max') == 'M') & (df_source['reward'] == 1)
    df_source['missReward'] = conditions_missReward.astype(int)
    df_source.loc[(df_source['missReward'] == 1)]
    
    
    """
    Define free reward: unexpected reward after a correct reject
    """
    conditions_freeReward = (reIndex['outcome'].transform('max') == 'FR') & (df_source['reward'] == 1)
    df_source['freeReward'] = conditions_freeReward.astype(int)
    df_source.loc[(df_source['freeReward'] == 1)]
    
    
    """
    Confirm that the unique trials are equal to the expected numbers
    """
    #filtered_df = df_source[df_source['lick_3C']==1]
    #unique_trial_numbers = pd.unique(filtered_df.index.get_level_values('TrialNumber'))
    #unique_trial_numbers
    #print(len(unique_trial_numbers))
    
    #this is the updated data frame now with all the behavioral events of interest included.
    return (df_source)
    


