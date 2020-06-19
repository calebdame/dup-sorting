import numpy as np
import pandas as pd
import operator

#
# df: pandas dataframe OR filename and path to read in
#     column structure must be [ID_a, ID_b, weight/probability]
# start_col: start alg in column 0 or 1
#            if 2, then both are done and the intersection of the two is 
#            returned (takes twice as long)
# return_dups: TRUE // returns the set of all duplicates after the sorted set
#              FALSE // returns just the sorted set 
# dups: (bool) if the dataframe only contains pairs where at least one of the
#              years is found in another prediction set
# 
# A Naive method for removing duplicates that keeps the highest 
# only highest probability matches for each involved ID.  For best results, 
# run starting from 
#

def naive_dedup(df, start_col=0, return_dups=False, dups = False):
    
    def dedup(d, index1):
        ark2 , prob = list(), list()
        for i in index1:
            di = d[i]
            if len(di) == 1:
                ark2 += [list(di.keys())[0]]
                prob += [list(di.values())[0]]
            else:
                k = max(di.items(), key=operator.itemgetter(1))[0]
                ark2 += [k]
                prob += [di[k]]
        return ark2, prob
    
    repeat = False
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(prediction_file)
    if start_col == 1:
        other_col = 0
    elif start_col == 0:
        other_col = 1
    elif start_col == 2:
        start_col = 0
        other_col = 1
        repeat = True
        
    if dups:
        just_dups = df
    else:
        a = df[df.columns[0]]
        b = df[df.columns[1]]
        g = df[a.isin(a[a.duplicated(keep=False)])]
        h = df[b.isin(b[b.duplicated(keep=False)])]
        just_dups = pd.concat([g,h],ignore_index=True).drop_duplicates(subset=[df.columns[0],df.columns[1]])
        just_dups = just_dups.loc[:, ~just_dups.columns.str.contains('^Unnamed')]

    index1 = list(set(just_dups.iloc[:,start_col]))
    d = just_dups.groupby(just_dups.columns[start_col])[[just_dups.columns[other_col],just_dups.columns[2]]].apply(lambda x: dict(x.values)).to_dict()
    ark2, prob = dedup(d, index1)
    
    if start_col == 0:
        first = pd.DataFrame(data=dict(zip(just_dups.columns, [index1, ark2, prob])))
    else:
        first = pd.DataFrame(data=dict(zip(just_dups.columns, [ark2, index1, prob])))
    index2 = list(set(first.iloc[:,other_col]))
    d = first.groupby(first.columns[other_col])[[first.columns[start_col],'link_prob']].apply(lambda x: dict(x.values)).to_dict()
    ark1, prob = dedup(d, index2)

    if start_col:
        df = pd.DataFrame(data=dict(zip(just_dups.columns, [index2, ark1, prob])))
    else:
        df = pd.DataFrame(data=dict(zip(just_dups.columns, [ark1, index2, prob])))

    if not repeat:
        if return_dups:
            return df, just_dups
        else:
            return df
    else: 
        df2 = naive_dedup(just_dups, start_col = other_col, return_dups = False, dups = True)
        df3 = pd.merge(df, df2, how='inner', on=[list(df.columns)[0],list(df.columns)[1]])
        df3 = df3[[list(df3.columns)[0],list(df3.columns)[1],list(df3.columns)[2]]]
        df3.columns = list(df.columns)
        if return_dups:
            return df3, just_dups
        else:
            return df3