import pandas as pd
import re

def extract(files):
    dfs = []
    for fname in files:
        dfs.append(pd.read_csv(fname))
    all_data = pd.concat(dfs, ignore_index=True)
    num_spam = (all_data.label == 1).sum()
    num_ham = (all_data.label == 0).sum()
    print(f"The number of spam emails is {num_spam}, and ham emails is {num_ham}", flush=True)
    all_data['text_combined'] = all_data['subject'].fillna('') + ' ' + all_data['body'].fillna('')
    all_data['text_length'] = all_data['text_combined'].apply(len)
    all_data['has_url'] = all_data['text_combined'].apply(lambda x: int(bool(re.search(r'http[s]?://|www\.', str(x)))))
    final_cols = ['text_combined', 'text_length', 'has_url', 'label']
    all_data = all_data[final_cols]
    return all_data
