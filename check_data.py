'''
Some test functions for checking downloaded data
'''

import pandas as pd

CHECK_AMTS = {'alj_id': 6,
             'dab_id': 4,
             'decision': 50,
             'dab_text': 1000,
             'alj_text': 1000,
             'alj_url': 50,
             'dab_url': 0}

def check_all(df, known_col='dab_url', check_dict=CHECK_AMTS):
    '''
    Checks created df for viable entries

    Inputs:
        df: dataframe to be checked
        known_col: column in original df known to have a non-null value for all obs.
        check_dict: a dictionary of the min feasible lengths for each col

    Output: a df with issues for each known_col (dab_url) noted
    '''
    # initial df
    issue_df = pd.DataFrame(df[known_col])
    for col in df.columns:
        min_amt = check_dict[col]
        bad = df[df[col].apply(lambda x: len(str(x)) < min_amt)][known_col]
        tmp = pd.DataFrame(bad)
        tmp[col] = f'bad_{col}'
        issue_df = pd.merge(issue_df, tmp, how='outer') 
    return issue_df.dropna(thresh=2)
