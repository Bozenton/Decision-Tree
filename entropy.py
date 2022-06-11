import numpy as np
import pandas as pd

def entropy(pmf, epsilon=1e-4):
    try:
        pmf = np.array(pmf)
        mask = pmf>epsilon
        pmf = pmf[mask]  # 0log0 = 0
        return -1.0 * np.dot(pmf, np.log2(pmf))
    except TypeError as te:
        print(pmf, "is not iterable")


def conditionalEntropy(cond_p, pmfs):
    assert len(cond_p) == len(pmfs)
    result = 0
    for i, p in enumerate(cond_p):
        result += p*entropy(pmfs[i, :])
    return result
    

def empiricalEntropy(label: pd.Series):
    pmf = label.value_counts().values / label.count()
    return entropy(pmf)

def empiricalConditionalEntropy(label: pd.Series, feature: pd.Series):
    df = pd.DataFrame({'label':label, 'feature': feature})
    feature_values = feature.unique()  # get all distinct feature values
    result = 0
    for i, value in enumerate(feature_values):
        df_filtered = df.query('feature==@value')   # filter by feature value
        label_filtered = df_filtered['label']
        ee = empiricalEntropy(label_filtered)
        # condtitional possibility
        cond_p = feature.value_counts()[value]/feature.count()
        result += cond_p*ee
    return result

if __name__ == '__main__':
    # test 
    df = pd.read_csv('./test_data.csv', delimiter='\t')
    label = df['type']
    empiricalEntropy(label)
    empiricalEntropy(label) - empiricalConditionalEntropy(label=df['type'], feature=df['age'])      # 0.083
    empiricalEntropy(label) - empiricalConditionalEntropy(label=df['type'], feature=df['work'])     # 0.324 
    empiricalEntropy(label) - empiricalConditionalEntropy(label=df['type'], feature=df['house'])    # 0.420
    empiricalEntropy(label) - empiricalConditionalEntropy(label=df['type'], feature=df['credit'])   # 0.363
    
