# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 20:17:14 2025

@author: AndreasWombacher
"""

import pandas as pd
import random
import shutil

# copy data to the training and the valuation dataset
#
# | Class    | Train   | Validation |
# | -------- | ------- | ---------- |
# | bird     | 300–600 | 80–150     |
# | not_bird | 300–600 | 80–150     |

base_dir = "jonathan/"

df = pd.read_excel(base_dir+"output/file_metadata.xlsx", parse_dates=['creation_time'])

df.columns
df.dtypes

max_index = df[df['Unnamed: 5'].notnull()].index[0]
df = df[df.index<max_index]
df['train'] = 0
df['val'] = 0

df['date'] = df['creation_time'].apply(lambda x: x.date()) 
df['hour'] = df['creation_time'].apply(lambda x: x.hour) 
df['number'] = df['filename'].apply(lambda x: x[6:-4]).astype(int)
df['number_rnd'] = df['number']//1000

df_agg = df.groupby(['Bird']).size().rename('cnt').reset_index()
df_agg2 = df.groupby(['number_rnd','Bird']).size().rename('cnt').reset_index()


df_agg = df.groupby(['train','val','Bird']).size().rename('cnt').reset_index()

# 300 training
# 75 validation
# 1 out of 5 is validation

bird = df.loc[df['Bird']==1].reset_index()
bird['label'] = (bird.index % 5)
bird['train'] = bird['label'].apply(lambda x: x>0)
bird['val'] = bird['label'].apply(lambda x: x==0)

non = df.loc[df['Bird']==0].reset_index()
ret = []
for ind, row in df_agg2.iterrows():
    dd = non[non['number_rnd'] == row['number_rnd']].reset_index()
    if len(dd)>0:
        numbers = random.sample(range(0, len(dd)), 8)
        print(numbers)
        for number in numbers:
            ret.append(dd.iloc[number])
non_bird = pd.DataFrame(ret)
non_bird = non_bird.drop('level_0', axis=1)
non_bird = non_bird.drop('index', axis=1)
non_bird = non_bird.reset_index()

non_bird['label'] = (non_bird.index % 5)
non_bird['train'] = non_bird['label'].apply(lambda x: x>0)
non_bird['val'] = non_bird['label'].apply(lambda x: x==0)

#%%
# move files for birds

files = bird[bird['train']]['filename'].to_list()
for f in files:
    shutil.move(base_dir+"images/"+f, base_dir+"dataset/train/bird/"+f)

files = bird[bird['val']]['filename'].to_list()
for f in files:
    shutil.move(base_dir+"images/"+f, base_dir+"dataset/val/bird/"+f)

#%%
# move files for birds

files = non_bird[non_bird['train']]['filename'].to_list()
for f in files:
    try:
        shutil.move(base_dir+"images/"+f, base_dir+"dataset/train/not_bird/"+f)
    except:
        print(f"failed for {f}")

files = non_bird[non_bird['val']]['filename'].to_list()
for f in files:
    try:
        shutil.move(base_dir+"images/"+f, base_dir+"dataset/val/not_bird/"+f)
    except:
        print(f"failed for {f}")
