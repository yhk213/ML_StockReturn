
# For Data Cleansing 
# SeongDeok Ko - 2020-38259 
#%% Import Module

import numpy as np
import pandas as pd 
from collections import Counter
from google.colab import drive
drive.mount('/content/drive')
# permno, DATE ,age,convind , divi, divo ,ps, securedind ,sin ,nincr, sic2 , ms : integer

#%% Data Loading Xiu Data

firm_level_data = pd.read_csv('/content/drive/MyDrive/datashare.csv')

# sorting with 'date' value
firm_level_data[firm_level_data.columns[2:]] = firm_level_data[firm_level_data.columns[2:]].astype('float32')
firm_level_data = firm_level_data.sort_values(by = ['DATE', 'permno'], ascending = True)

firm_level_data[firm_level_data.columns[0:2]] = firm_level_data[firm_level_data.columns[0:2]].astype(np.int32) 

date = firm_level_data['DATE']
result = Counter(date)
date_firm = list(result.keys())
date_num_firm = list(result.values())

del date
del result

#%% Goyal and Welch Data
goyal = pd.read_excel('/content/drive/MyDrive/PredictorData2019.xlsx',sheet_name='Monthly')

goyal = goyal.iloc[1034:1752,:]   #From 195703~ 201612
num_macro = 8

macro_data = np.full((goyal.shape[0],num_macro),np.nan, dtype = np.float32 )

macro_data[:,0] = np.log(goyal['D12'] / goyal['Index'])  # d/p
macro_data[:,1] = np.log(goyal['E12'] / goyal['Index'])  # e/p
macro_data[:,2] = goyal['b/m']                   # B/M
macro_data[:,3] = goyal['ntis']                  # net equity expansion
macro_data[:,4] = goyal['tbl']                   # Treasury-bill rate
macro_data[:,5] = goyal['lty'] - goyal['tbl']    # term-spread
macro_data[:,6] = goyal['BAA'] - goyal['AAA']    # default-spread
macro_data[:,7] = goyal['svar']                  # stock variance

col_macro = ['d/p', 'e/p', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

macro_data = pd.DataFrame(data = macro_data, columns = col_macro) # size : 718*8
macro_data = macro_data.astype('float32')

#%% Cleansing Start 
# Step 1
# Return (Y variable) from  CRSP - HPR monthly (1957.4 ~ 2017.1) 
# match Y variable to X
# if X : 1957.03, search the return of firm in 1957.04   (using for-loop to find permno)
#   ...
#   ...
# if X : 2016.12, search the return of firm in 2017.01 


ret = pd.read_csv('/content/drive/MyDrive/hpr_total_crsp.csv')             # size : 4,030,269 * 3 
ret = ret.sort_values(by=['date','PERMNO'], ascending = True)
print(ret.shape)

# delete rows that have 'nan' / 'B' / 'C' in 'RET' column
ret = ret.dropna(axis=0)
ret = ret[ret.RET != 'B']
ret = ret[ret.RET != 'C']
#print(ret.shape)                                 # size : 3,864,208 * 3

date_= ret['date']
result_ = Counter(date_)
date_ret = list(result_.keys())
date_num_ret = list(result_.values())

del date_
del result_


#%% Step 2 :
# HPR > Excess return

tbl_rate = goyal['tbl']
tbl_rate_concat = []

for i in range(len(date_ret)):
    for j in range(date_num_ret[i]):
        tbl_rate_concat.append(tbl_rate.iloc[i]*(30/365))   # monthly

tbl_rate_concat = np.asarray(tbl_rate_concat).reshape(-1,1)
tbl_rate_concat = tbl_rate_concat.astype('float32')
ret_concat = ret['RET'].to_numpy(dtype='float32').reshape(-1,1)   

excess_ret = ret_concat - tbl_rate_concat
excess_ret = pd.DataFrame(data = excess_ret, columns = ['excess_ret'] )
 
ret = pd.concat([ret.iloc[:,:2].reset_index(drop=True), excess_ret.reset_index(drop=True)], axis=1)  # size : 3,864,208 * 3


del tbl_rate
del tbl_rate_concat
del ret_concat
del excess_ret

#%% Step 3
for i in range(len(date_firm)):   
    match_ret = np.full((date_num_firm[i],1),np.nan)   
    
    if i==0:
        sum_stock = 0
        sum_ret = 0
    else: 
        sum_stock = sum(date_num_firm[:i])
        sum_ret = sum(date_num_ret[:i])  
    
    last_index = 0
    for j in range(date_num_firm[i]):
        print(i,j)
        for k in range(last_index, date_num_ret[i]):
            if firm_level_data.iloc[sum_stock+j,0] == ret.iloc[sum_ret+k,0]:
                match_ret[j,0] = ret.iloc[sum_ret+k,2]
                last_index = k+1 
                break
    
    if i ==0:
        matched_ret = match_ret
    else: 
        matched_ret = np.concatenate((matched_ret, match_ret), axis = 0)
        
# size 3,760,208*3
matched_ret = pd.DataFrame(data = matched_ret, columns = ['excess_return'])
matched_ret = pd.concat([firm_level_data.iloc[:,:2].reset_index(drop=True), matched_ret.reset_index(drop=True)], axis=1)

del match_ret
del last_index
del sum_stock
del sum_ret

#%% Step 4 
# fill Na in X 
# Using All data years, size : 3,760,208*97,  97 = permno, date, 94variables, SIC-code 
# Missing values: use cross-sectional median at each month 

# for missing values 
for i in range(len(date_firm)):
    print(i)
    if i==0:
        sum_firm = 0
    else:
        sum_firm = sum(date_num_firm[:i])
    
    data = firm_level_data.iloc[sum_firm:(sum_firm + date_num_firm[i]), :]
    
    # except (permno, Date, Sic2), replace 'nan' with median   
    for j in range(2,96):
        replacing = data.iloc[:,j].fillna(data.iloc[:,j].median())
        
        if j==2:
            new_data = pd.concat([data.iloc[:,:2], replacing], axis=1)
            #print(new_data.shape)
        else:
            new_data = pd.concat([new_data, replacing], axis=1)
    
    # concat sic2 column
    new_data = pd.concat([new_data, data.iloc[:,-1]], axis=1)
    #print(new_data.shape)
    
    if i==0:
        new_firm_data = new_data
    else:
        new_firm_data = pd.concat([new_firm_data, new_data], axis=0)

# still exist nan value ... > 0  (ex 1957.3  some values are all nan for all firms)
new_firm_data = new_firm_data.fillna(0)


del firm_level_data
del replacing
del new_data
del sum_firm
del data
  
num_firm_charac = 94

#%% Step 5
# concat X, y and delete row if y has Nan 
# and divide X,y again 

new_firm_data = pd.concat([new_firm_data.reset_index(drop=True), matched_ret.iloc[:,-1].reset_index(drop=True)], axis=1)
new_firm_data = new_firm_data.dropna(axis=0)    # size : 3,709,909 * (97+1)
print(new_firm_data.shape)


# size has changed.. 
date = new_firm_data['DATE']
result = Counter(date)
date_firm = list(result.keys())
date_num_firm = list(result.values())


matched_ret = new_firm_data.iloc[:,-1]
new_firm_data = new_firm_data.iloc[:,:-1]

del ret
del date
del result



#%% Step 6
# last column 'sic2' : 74 SIC number > change to dummy variable  (3,760,208 * 74) 

sic2 = new_firm_data['sic2'].to_numpy()

sic_dummy = np.full((new_firm_data.shape[0],74),0, dtype = np.float32)  # 74 sic code

for i in range(new_firm_data.shape[0]):
    print(i)
    for j in range(1,75):
        if sic2[i]==j:
            sic_dummy[i,j-1]=1

# for naming columns
sic_col = [] 
for i in range(1,75):
    sic_col.append('sic_dummy'+str(i))

sic_dummy = pd.DataFrame(data = sic_dummy, columns = sic_col)
sic_dummy = sic_dummy.astype('int8')

# 3,760,208 * (96+74)
new_firm_data = pd.concat([new_firm_data.iloc[:,:-1].reset_index(drop=True), sic_dummy.reset_index(drop=True)], axis=1)

del sic2
del sic_col
del sic_dummy

new_firm_data = pd.concat([new_firm_data.reset_index(drop=True), matched_ret.reset_index(drop=True)], axis=1)
new_firm_data.to_csv('/content/drive/MyDrive/x_y_wo_inter.csv', index = False)

del matched_ret


#%% Step 7 : Temporary 
#  Goyal 718*8  > 3,760,208*8 (to make intersection)

macro_data_resized = np.full((new_firm_data.shape[0], macro_data.shape[1]), 0, dtype = np.float32)

t=0
for i in range(len(date_num_firm)):
    print(i)
    for j in range(date_num_firm[i]):
        macro_data_resized[t,:] = macro_data.iloc[i,:]
        t=t+1

macro_data_resized = macro_data_resized.astype('float32')
del macro_data
del goyal


#%% Step 8 : Make Interaction

for i in range(macro_data_resized.shape[1]):
    print(i)
    if i==0:    
        # new firm data permn, date, 94, 74, excess return
        interact = new_firm_data.iloc[:,2:-75] * (macro_data_resized[:,i].reshape(-1,1))
        #print(interact.shape)
    else:
        interact = np.concatenate((interact, new_firm_data.iloc[:,2:-75] * (macro_data_resized[:,i].reshape(-1,1))), axis=1)

# make column names for interact term
col_stock = list(new_firm_data.columns[2:-75]) 

col_interact = []
for i in col_macro:
    for j in col_stock:
        col_interact.append(str(j)+'_'+str(i))
        
interact = pd.DataFrame(data = interact, columns = col_interact)
interact = pd.concat([new_firm_data.iloc[:,:2].reset_index(drop=True), interact.reset_index(drop=True)], axis=1)
interact.to_csv('/content/drive/MyDrive/interact.csv', index = False)

del col_stock
del col_interact
