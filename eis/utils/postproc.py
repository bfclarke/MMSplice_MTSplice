import pandas as pd
import numpy as np

def max_varEff(df, combine_fn=lambda x: np.sum(x, axis=1)):
    """ Summarize largest absolute effect per variant across all affected exons.
    Args:
        df: result of `predict_all_table`
        combine_fn: Maximum effect size is calculated by calculating 
        the combined effect size of all 5 modules.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)
    ref_list = ['EIS_ref_acceptorIntron', 'EIS_ref_acceptor', 'EIS_ref_exon', 'EIS_ref_donor', 'EIS_ref_donorIntron']
    alt_list = ['EIS_alt_acceptorIntron', 'EIS_alt_acceptor', 'EIS_alt_exon', 'EIS_alt_donor', 'EIS_alt_donorIntron']
    EIS_diff = df[alt_list].values - df[ref_list].values
    df['EIS_diff'] = combine_fn(EIS_diff)
    dfMax = df.groupby(['ID'], as_index=False).agg({'EIS_diff': lambda x: max(x, key=abs)})
    dfMax = dfMax.merge(df, how='left', on = ['ID', 'EIS_diff'])
    dfMax = dfMax.drop_duplicates(subset=['ID', 'EIS_diff'])
    # dfMax = dfMax.drop("EIS_diff", axis=1)
    return dfMax

def _not_close0(arr):
    return ~np.isclose(arr, 0)

def _transform(X, region_only=False):
    ''' Make interaction terms for the overlapping prediction region
    Args:
        X: modular prediction. Shape (, 5)
        region_only: only interaction terms with indicator function on overlapping
    '''
    exon_overlap = np.logical_or(np.logical_and(not_close0(X[:,1]), not_close0(X[:,2])), np.logical_and(not_close0(X[:,2]), not_close0(X[:,3])))
    acceptor_intron_overlap = np.logical_and(not_close0(X[:,0]), not_close0(X[:,1]))
    donor_intron_overlap = np.logical_and(not_close0(X[:,3]), not_close0(X[:,4]))

    if region_only:
        X = np.hstack([X, (X[:,2]*exon_overlap).reshape(-1,1)])
        X = np.hstack([X, (X[:,4]*donor_intron_overlap).reshape(-1,1)]) 
        X = np.hstack([X, (X[:,0]*acceptor_intron_overlap).reshape(-1,1)])
    else:
        X = np.hstack([X, (X[:,2]*exon_overlap).reshape(-1,1)])
        X = np.hstack([X, (X[:,4]*donor_intron_overlap).reshape(-1,1)]) 
        X = np.hstack([X, (X[:,0]*acceptor_intron_overlap).reshape(-1,1)])
    
    return X