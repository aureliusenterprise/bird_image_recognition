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

df = pd.read_excel(base_dir+"output/file_metadata2.xlsx", parse_dates=['creation_time'])

df.columns
df.dtypes

df = df.iloc[1:-2].copy()

df['date'] = df['creation_time'].apply(lambda x: x.date()) 
df['hour'] = df['creation_time'].apply(lambda x: x.hour) 
df['number'] = df['filename'].apply(lambda x: x[6:-4]).astype(int)
df['number_rnd'] = df['number']//1000

df_agg = df.groupby(['Bird']).size().rename('cnt').reset_index()
df_agg2 = df.groupby(['number_rnd','Bird']).size().rename('cnt').reset_index()

# detected as birds, but are not birds
# add these as negative cases for training and validation
#df_agg = df.groupby(['Train','Val','Bird']).size().rename('cnt').reset_index()
df2_train = df[df['Train']==1].copy()
df2_val = df[df['Val']==1].copy()

# add negative files
# move files for non birds

files = df2_train['filename'].to_list()
for f in files:
    try:
        shutil.move(base_dir+"images/"+f, base_dir+"dataset/train/not_bird/"+f)
    except:
        print(f"failed for {f}")

files = df2_val['filename'].to_list()
for f in files:
    try:
        shutil.move(base_dir+"images/"+f, base_dir+"dataset/val/not_bird/"+f)
    except:
        print(f"failed for {f}")

#%%
# classified as birds and are indeed birds
# add these as positive cases for training 
df2_bird = df[df['Bird']==1].copy()

files = df2_bird['filename'].to_list()
for f in files:
    shutil.move(base_dir+"images/"+f, base_dir+"dataset/train/bird/"+f)

# all data from non birds are anyhow birds.
# add these as positive cases
bird = pd.read_csv(base_dir+"output/file_metadata2_non.csv", parse_dates=['creation_time'])

# 1 out of 5 is validation
bird['label'] = (bird.index % 5)
bird['train'] = bird['label'].apply(lambda x: x>0)
bird['val'] = bird['label'].apply(lambda x: x==0)

# move files for birds

files = bird[bird['train']]['filename'].to_list()
for f in files:
    shutil.move(base_dir+"images/non/"+f, base_dir+"dataset/train/bird/"+f)

files = bird[bird['val']]['filename'].to_list()
for f in files:
    shutil.move(base_dir+"images/non/"+f, base_dir+"dataset/val/bird/"+f)

