import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csgraph
from scipy.optimize import linear_sum_assignment as lsa
import itertools

#
# df: pandas dataframe OR filename and path to read in
# 
# return_dups: TRUE // returns the set of all duplicates after the sorted set
#              FALSE // returns just the sorted set 
# dups: (bool) if the dataframe only contains pairs where at least one of the
#              years is found in another prediction set
#
# A Linear assignment solver for the many networks of duplicates that overlap
# maximizing the sum of prediction probabilities/weights across possible matches 
# using the Hungarian / Munkres assignment algorithm 
#

def hung_dedup(df, return_dups=False, dups = False):
    
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(prediction_file)
    if dups:
        just_dups = df
    else:
        a = df[df.columns[0]]
        b = df[df.columns[1]]
        g = df[a.isin(a[a.duplicated(keep=False)])]
        h = df[b.isin(b[b.duplicated(keep=False)])]
        just_dups = pd.concat([g,h],ignore_index=True).drop_duplicates(subset=[df.columns[0],df.columns[1]])
        just_dups = just_dups.loc[:, ~just_dups.columns.str.contains('^Unnamed')]
        
    index1, index2 = list(set(just_dups.iloc[:,0])), list(set(just_dups.iloc[:,1]))
    col1, col2, col3 = list(just_dups.iloc[:,0]), list(just_dups.iloc[:,1]), list(just_dups.iloc[:,2])

    k = len(index1)
    ind_ind1 = dict(zip(index1,list(range(k))))
    j = k + len(index2)
    ind_ind2 = dict(zip(index2,list(range(k,j))))

    inv_ind1 = {v: k for k, v in ind_ind1.items()}
    inv_ind2 = {v: k for k, v in ind_ind2.items()}

    row, col, data = [], [], []

    row = [ind_ind1[x] for x in col1]
    col = [ind_ind2[x] for x in col2]
    data = list(-1*np.array(just_dups.iloc[:,2]))
    A = csc_matrix((data,(row,col)),shape=(j,j))
    
    jl = list(range(j))
    n , f = csgraph.connected_components(A, directed=False,connection='weak')
    l = {k : set(map(lambda x: x[1],v)) for k, v in itertools.groupby(sorted(zip(f,jl)),key=lambda x: x[0])}

    pairs = list(zip(row,col))
    get_prob = dict(zip(pairs,data))
    num_connected_sets = len(l)
    y1ind, y2ind = set(range(k)), set(range(k,j))
    pairs = set(pairs)
    year1, year2, prob = [],[],[]

    for i in range(num_connected_sets):
        s = l[i]
        y1 = list(s & y1ind)
        y2 = list(s & y2ind)
        mat = np.zeros((len(y1),len(y2)))
        q = itertools.product(y1,y2)
        for a in q:
            if a in pairs:
                one, two = a
                mat[y1.index(one),y2.index(two)] = get_prob[a]
        r, c = lsa(mat)
        for j in range(len(c)):
            a = (y1[r[j]],y2[c[j]])
            if a in pairs:
                year1 += [inv_ind1[y1[r[j]]]]
                year2 += [inv_ind2[y2[c[j]]]]
                prob += [get_prob[a]]
    prob = list(-1*np.array(prob))
    final = pd.DataFrame(list(zip(year1,year2,prob)), columns=[just_dups.columns[0],just_dups.columns[1],'link_prob'])
    if return_dups:
        return final, just_dups
    else:
        return final
