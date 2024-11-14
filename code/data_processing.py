# read in data and filter ROI relavant to thickness and Volumn

import numpy as np
from numpy import linalg
# from numpy._core.numeric import identity: with more recent numpy > 2.0 ? 
from numpy.core.numeric import identity
import pandas as pd
import csv
import itertools
import statsmodels.regression.mixed_linear_model as sm
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

DATAPATH = 'data/irene-phd/ukb-data'
datadir = os.path.join(os.path.expanduser('~'),DATAPATH)
# need to check if datadir exists and if not create it
# if not os.path.exists(datadir):
#    os.path.mkdir(datadir)
os.chdir(datadir)

# %%
#iter_csv = pd.read_csv('ukb47552.csv', iterator=True, chunksize=10000,error_bad_lines=False)
iter_csv = pd.read_csv('ukb47552.csv', iterator=True, chunksize=10000,on_bad_lines='error')
# this csv contains all the imaging features
data = pd.concat([chunk.dropna(how='all') for chunk in iter_csv] )
data.shape #502462 subjects, 2545-1 measurements (column 0 is eid)  

# %%
# filter data by the field id related to "thickness" and "Volumn"
url = "https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=196"
field_id = pd.DataFrame(pd.read_html(url)[0])
# Irene: still having problem with read_html, error: lmxl is not installed (but I definitely have it installed)
# get the field description that have volume or thickness, then take the corresponding row numbers and make them str
rows = [s for s, x in enumerate(field_id["Description"]) if "thickness" in x or "Volume" in x]
filtered_id = field_id.loc[rows]["Field ID"].apply(str)

# %%
iter_csv = pd.read_csv('ukb46307_demographics.csv', iterator=True, chunksize=10000,on_bad_lines='error')
data_demographics = pd.concat ([chunk.dropna(how='all') for chunk in iter_csv] )
data_demographics.shape # 502462 subjects, 11 measurements
                        # JB : finds (502485, 11)
data_demographics = data_demographics[['eid','age_at_recruitment', 'sex']]
# %%
merge_data = data.merge(right = data_demographics, how="inner", on="eid")
merge_data.shape
# JB : len(merge_data) : 502462
# merge_data contains image features and demographics with all rows

# %%
# extract more phenotypes
# ukb field 20544 #2(SCZ), #12(Autism), #14(ADD/ADHD)
#     field 20191 fluid intelligence at

column_codes = {
    "eid":"eid",
    "21003-2.0": "age_at_ses2",
    "20191-0.0":"fluid intelligence",
    "20544-0.2":"scz",
    "20544-0.12":"autism",
    "20544-0.14":"adhd"}

phenotype = pd.read_csv("current.csv", usecols=column_codes.keys())
phenotype = phenotype.dropna(axis=0, how="all")
phenotype.shape
# 6801 subjects in the phenotype data

# %%
# merge_data2 : has imaging features + demographics + phenotypes
merge_data2 = merge_data.merge(right=phenotype, how="inner", on="eid")
merge_data2.shape
#(6801, 2552)
# merge_data : the first 2545 columns are image features, the last 2 columns are age and sex
# merge_data2 : adds the 6 columns of phenotypes to the right 
# merge_data2 : remove rows that are not in phenotype.csv (csv provided by Nikhil)

print(merge_data.shape)
assert(merge_data.shape==(502462, 2547))
print(merge_data2.shape)
assert(merge_data2.shape==(6801,2552))
# JB: maybe change column names of merge_data2 with the "column_codes" before getting the traits below 
traits = merge_data2[['eid', 'age_at_recruitment', 'sex', '20191-0.0', '20544-0.2', '20544-0.12','20544-0.14', '21003-2.0']]
print(traits.shape)
assert(traits.shape==(6801,8))
# 

# JB: some quick comment here ? why 2545 in merge_data? 
# now, we take traits (8 columns ?) and create 2 DF : instance2 and instance3 (visit 2 and 3), 
# these share the same demo and phenotypes but not imaging feature
instance2_dat = instance3_dat = pd.DataFrame(traits)

nb_of_imaging_feature_col = data.shape[1]
assert(nb_of_imaging_feature_col==2545)

# merge_data 
for col in merge_data2.columns[1:nb_of_imaging_feature_col]: #remove the eid on column 0
    field, instance = col.split("-")
    if any(filtered_id == field):
        if instance == "2.0":
            instance2_dat[field] = data[col]
        if instance == "3:0":
            instance3_dat[field] = data[col]

col_to_drop = ['eid', 'age_at_recruitment', 'sex', '20191-0.0', '20544-0.2', '20544-0.12','20544-0.14', '21003-2.0']

instance2_dat.drop(columns=col_to_drop, axis=1)
instance3_dat.drop(columns=col_to_drop, axis=1)

instance2_dat.dropna(axis = 0, how = "all", inplace=True)
instance3_dat.dropna(axis = 0, how = "all", inplace=True)

# Consider only complete cases, each instances have 6801 subjects with 132 features (including morphological measurs and phenotypes)
print(instance2_dat.shape, instance3_dat.shape)
# Consider only complete cases, each instances have 43107 subjects with 62
# JB: why the print is returning:  
#:  print(instance2_dat.shape, instance3_dat.shape) 
#: (6801, 132) (6801, 132)

# Take a subset of instance2 first
# sub_dat is only used for testing the code
sub_dat = instance2_dat.iloc[0:1000]

# write 3 files in the current directory : sub_dat.csv, ses2.csv and ses3.csv
# os.chdir(datadir)
# out = os.getcwd()

# ses2 and ses3 are the csv with imaging and other phenotype for ses 2 and 3
df_map = {'sub_dat': sub_dat,'ses2': instance2_dat, 'ses3': instance3_dat}
for name, df in df_map.items():
    df.to_csv(os.path.join(datadir, f'{name}.csv'))

# %%
# application data:

