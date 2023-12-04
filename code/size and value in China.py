# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 16:51:38 2023

@author: Chenx
"""

import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import datetime
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir(r'D:\Python\Code\Spyder\ChinaAShareEquityCharacteristics-main')

# load Month data
data_Month = pd.read_csv('data/csmar_tables/TRD_Mnth.csv', 
                         usecols=['Stkcd', 'Trdmnt', 'Msmvosd', 'Msmvttl', 'Mretwd', 'Markettype', 'Ndaytrd'])
data_Month.Trdmnt = data_Month.Trdmnt.apply(lambda x: x[0:4] + x[5:7])
data_Month = data_Month[(data_Month.Trdmnt >= '199901') & (data_Month.Trdmnt <= '201712')]
data_Month.index = range(len(data_Month))
Trdmnt = pd.DataFrame(data_Month.Trdmnt.unique()).rename(columns={0:'Trdmnt'}).sort_values(by='Trdmnt')


##################################
#           functions            #
##################################

def data_Dalyr():
    
    data_Dalyr = pd.read_csv('data/csmar_tables/TRD_Dalyr.csv', 
                             usecols=['Stkcd', 'Trddt', 'Dretwd', 'Markettype'])
    data_Dalyr = data_Dalyr[data_Dalyr.Markettype.isin([1, 4, 16])]
    data_Dalyr.Trddt = data_Dalyr.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:])
    data_Dalyr = data_Dalyr[(data_Dalyr.Trddt >= '19990101') & (data_Dalyr.Trddt <= '20171231')]
    data_Dalyr.index = range(len(data_Dalyr))
    
    return data_Dalyr

def get_stock_groups(data, sortname, groups_num):

    df = data.copy()
    labels=['bottom', 'top']
    try:
        groups = pd.DataFrame(pd.qcut(df[sortname], groups_num, 
                                      labels=labels).astype(str)).rename(columns={sortname: 'Group'})
    except:
        groups = pd.DataFrame(pd.qcut(df[sortname].rank(method='first'), groups_num, 
                                      labels=labels).astype(str)).rename(columns={sortname: 'Group'})

    return groups

# Double Sort
def Double_Sort(data, SortMethod, TimeName, sortname1, groups_num1, labels1, 
                sortname2, groups_num2, labels2):
    
    df = data.copy()
    # Step 1 - Single Sort
    def get_stock_groups(data, sortname, groups_num, labels):

        df = data.copy()
        try:
            groups = pd.DataFrame(pd.qcut(df[sortname], groups_num, 
                                          labels=labels).astype(str)).rename(columns={sortname: 'Group'})
        except:
            groups = pd.DataFrame(pd.qcut(df[sortname].rank(method='first'), groups_num, 
                                          labels=labels).astype(str)).rename(columns={sortname: 'Group'})
    
        return groups
  
    PortTag1 = df.groupby([TimeName]).apply(get_stock_groups, sortname1, groups_num1, labels1)
    df = pd.merge(df, PortTag1['Group'], left_index=True, right_index=True)
    df.rename(columns={'Group': 'G1'}, inplace=True)
    
    # Step 2 - Independet or Dependet Sort
    if SortMethod == 'Independent':
        PortTag2 = df.groupby([TimeName]).apply(get_stock_groups, sortname2, groups_num2, labels2)
    else:
        PortTag2 = df.groupby([TimeName, 'G1']).apply(get_stock_groups, sortname2, groups_num2, labels2)
    df = pd.merge(df, PortTag2['Group'], left_index=True, right_index=True)
    df.rename(columns={'Group': 'G2'}, inplace=True)
    df['Group'] = df['G1'] + df['G2']

    return df 

def get_sort_result(data, TimeName, weighted):
    
    def cpt_vw_ret(group, avg_name, weight_name):
        
        d = group[avg_name]
        w = group[weight_name]
        try:
            return (d * w).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan  

    df = data.copy()
    df['Weight'] = df[weighted] / df.groupby([TimeName, 'Group'])[weighted].transform('sum') 
    ret_name = 'Mretwd'
    vwret = df.groupby([TimeName, 'Group']).apply(cpt_vw_ret, ret_name, 'Weight').to_frame().reset_index().rename(columns={0: 'Ret'})    
    vwret = vwret.set_index(TimeName) 
    
    return vwret   


##################################
#         calculate CH-3         # 
##################################

def data_processing(data):
    
    data_Month = data.copy()
    # 收益率滞前一期、股票筛选
    data_Month = data_Month[data_Month.Markettype.isin([1, 4, 16])]
    data_Month['Mretwd'] = data_Month.groupby('Stkcd').Mretwd.shift(-1)
    data_Month.dropna(subset=['Mretwd', 'Msmvttl'], inplace=True)
        
    
    #####################################################
    # less than 15 trading records in the past month    # 
    
    # 处理 A股市场因春节假期，当月交易日往往不足15天
    data_Dal = data_Dalyr()
    Dretwd = pd.pivot(data_Dal, index='Trddt', columns='Stkcd', values='Dretwd')
    stk_list = list(Dretwd.columns)
    Dretwd.reset_index(inplace=True)
    Dretwd['Trdmnt'] = Dretwd.Trddt.apply(lambda x: x[0:6])
    index = Dretwd.groupby('Trdmnt').tail(1).index
    Dretwd.drop(columns=('Trdmnt'), inplace=True)

    for i in stk_list:
        temp_data = Dretwd[[i]]
        for j in index:
            temp = temp_data.loc[j-19:j, :]
            not_nan_num = np.sum(~np.isnan(temp)).iloc[0]
            if not_nan_num < 15:
                Dretwd.loc[j, i] = np.nan
            
    Dretwd.set_index('Trddt', inplace=True)
    temp = Dretwd.unstack().reset_index()
    temp['Trdmnt'] = temp.Trddt.apply(lambda x: x[0:6])
    temp = temp.drop_duplicates(['Stkcd','Trdmnt'], keep='last')
    temp.dropna(subset=(0), inplace=True)
    data_Month = pd.merge(temp[['Stkcd', 'Trdmnt']], data_Month, on=['Stkcd', 'Trdmnt'], how='left')
    data_Month.dropna(subset=('Mretwd'), inplace=True)

    ############################################################
    # having less than 120 trading records in the past year    # 
    
    data_Month['trading_counts'] = np.nan
    stkcd = list(data_Month.Stkcd.unique())
    for i in stkcd:
        temp_data = data_Month[data_Month.Stkcd == i]
        if len(temp_data) >= 12:
            temp_data['month_counts'] = [i+1 for i in range(len(temp_data))]
            for j in range(12, len(temp_data)+1):
                temp = temp_data[(temp_data.month_counts >= j - 11) & (temp_data.month_counts <= j)]
                trading_counts = temp.Ndaytrd.sum()
                index = temp.index[-1]
                data_Month.loc[index, 'trading_counts'] = trading_counts
        else:
            data_Month.drop(index=temp_data.index, inplace=True)
    
    data_Month.drop(index=data_Month[data_Month.trading_counts <= 120].index, inplace=True)
    data_Month.drop(columns=['trading_counts'], inplace=True)
    
    ##################################
    # listed less than six months    # 
    
    # 股票上市不足半年 
    
    Listdt = pd.read_csv('data/csmar_tables/IPO_Cobasic.csv')
    Listdt.Listdt = Listdt.Listdt.apply(lambda x: x[:4] + x[5:7] + x[8:10])
    Listdt['Listdt_HalfYear'] = pd.to_datetime(Listdt.Listdt)
    Listdt['Listdt_HalfYear'] = Listdt['Listdt_HalfYear'].apply(lambda x: x + datetime.timedelta(days=182))
    Listdt.Listdt_HalfYear = Listdt.Listdt_HalfYear.astype('str')
    Listdt.Listdt_HalfYear = Listdt.Listdt_HalfYear.apply(lambda x: x[:4] + x[5:7])
    Listdt = Listdt[['Stkcd','Listdt_HalfYear']]
    
    data_Month = pd.merge(data_Month, Listdt, on='Stkcd', how='left')
    data_Month = data_Month[data_Month.Trdmnt > data_Month.Listdt_HalfYear]
    data_Month.drop(columns=('Listdt_HalfYear'), inplace=True)

    # having less than 120 trading records in the past year
    # data_Month['year'] = data_Month.Trdmnt.apply(lambda x: x[:4] )
    # trading_counts = data_Month.groupby(['year','Stkcd']).Ndaytrd.sum().reset_index().rename(columns={'Ndaytrd':'trading_counts'})
    # data_Month = pd.merge(data_Month, trading_counts, on=['Stkcd', 'year'], how='left')
    # data_Month = data_Month[data_Month.trading_counts >= 120]
    # data_Month.drop(columns=(['year', 'trading_counts']), inplace=True)
    
    # drop ST/PT
    # info_ST = pd.read_csv('data/csmar_tables/ST_PT.csv')
    # info_ST.Trdmnt = info_ST.Trdmnt.astype('str')
    # info_ST = info_ST[info_ST.Trdmnt <= '202111']
    # data_Month = data_Month.merge(info_ST, on=['Stkcd', 'Trdmnt'])
    # data_Month = data_Month[data_Month.Trdsta.isin([1, 4, 7, 10, 13])]
    
    
    ##################################
    # the bottom 30% of firm size    # 
    
    PortTag = data_Month.groupby(['Trdmnt']).apply(get_stock_groups, 'Msmvttl', [0, 0.3, 1.0])
    data_Month = pd.merge(data_Month, PortTag['Group'], left_index=True, right_index=True)
    data_Month = data_Month[data_Month.Group == 'top']
    data_Month.drop(columns=('Group'), inplace=True)
    
    return data_Month


# data processing
data_Month = data_processing(data_Month)


####################
# market factor    # 
####################

market_ttlvalue = data_Month.groupby('Trdmnt').Msmvttl.sum().reset_index().rename(columns={'Msmvttl':'market_ttlvalue'})
data_Month = pd.merge(data_Month, market_ttlvalue, on=['Trdmnt'], how='left')
data_Month['vwret'] = data_Month.Mretwd * data_Month.Msmvttl / data_Month.market_ttlvalue

MKT = data_Month.groupby('Trdmnt').vwret.sum().reset_index()

rf = pd.read_csv('data/csmar_tables/TRD_Nrrate.csv', usecols=['Clsdt', 'Nrrdata', 'Nrrmtdt'])
rf.rename(columns={'Clsdt':'Trdmnt'}, inplace=True)
rf.Trdmnt = rf.Trdmnt.apply(lambda x: x[:4] + x[5:7])
rf.Nrrdata = rf.Nrrdata / 100
rf.Nrrmtdt = rf.Nrrmtdt / 100
rf.drop_duplicates('Trdmnt', keep='last', inplace=True)

MKT = pd.merge(MKT, rf, on='Trdmnt', how='left')
MKT['MKTfactor'] = MKT.vwret - MKT.Nrrmtdt


####################
#     VMG   SMB    # 
####################

raw_data = pd.read_csv('data/csmar_tables/FS_Comins.csv')
raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])

raw_data.drop(index=raw_data[
    (raw_data.type != '1231') & (raw_data.type != '0930') & (raw_data.type != '0630') & (raw_data.type != '0331')].index, inplace=True)
raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])

# B002000000: 净利润
# B001400000：营业外收入
# B001500000：营业外支出

Net_profit = raw_data[['Stkcd', 'Accper', 'B002000000', 'B001400000', 'B001500000']]
Net_profit.index = range(len(raw_data))

lag_d = []
for i in range(Net_profit.shape[0]):
    if Net_profit.loc[i]['Accper'][4:] == '03':
        lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '04')
    if Net_profit.loc[i]['Accper'][4:] == '06':
        lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '07')
    if Net_profit.loc[i]['Accper'][4:] == '09':
        lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '10')
    if Net_profit.loc[i]['Accper'][4:] == '12':
        newyear = str(int(Net_profit.loc[i]['Accper'][0:4]) + 1)
        lag_d.append(newyear + '01')

Net_profit['Accper'] = pd.DataFrame(lag_d)
Net_profit.drop_duplicates(subset=['Stkcd','Accper'], keep='last', inplace=True)
Net_profit.rename(columns={'Accper':'Trdmnt'}, inplace=True)
Net_profit['Net_Profit_After_Ded_Nr_Lpnet'] = Net_profit.B002000000 - (Net_profit.B001400000 - Net_profit.B001500000)

# merge data
data_Month = pd.merge(data_Month, Net_profit[['Stkcd', 'Trdmnt', 'Net_Profit_After_Ded_Nr_Lpnet']], on=['Stkcd', 'Trdmnt'], how='left')
# data_Month = pd.merge(data_Month, Msmvttl_lag, on=['Stkcd', 'Trdmnt'], how='left')
data_Month.fillna(method='ffill', inplace=True)
universe = np.isnan(data_Month.Mretwd)
data_Month.Net_Profit_After_Ded_Nr_Lpnet[universe] = np.nan
# data_Month.Msmvttl_lag[universe] = np.nan

# calcuate ep 
data_Month['ep'] = data_Month.Net_Profit_After_Ded_Nr_Lpnet / (data_Month.Msmvttl * 1000)

data_Month = Double_Sort(data_Month, 'Independent', 'Trdmnt', 'Msmvttl', 2, ['S', 'B'],
                         'ep', [0, 0.3, 0.7, 1], ['G', 'M', 'V'])

result = get_sort_result(data_Month, 'Trdmnt', 'Msmvttl')

SV = result[result.Group == 'SV']['Ret']
BV = result[result.Group == 'BV']['Ret']
SG = result[result.Group == 'SG']['Ret']
BG = result[result.Group == 'BG']['Ret']

VMG = ((SV + BV) / 2) - ((SG + BG) / 2)

# VMG factor
VMG = pd.merge(Trdmnt, VMG, on='Trdmnt', how='left').rename(columns={'Ret':'VMG'})
VMG.VMG = VMG.VMG.shift(1)
VMG.VMG = VMG.VMG * 100

## SMB factor
SV = result[result.Group == 'SV']['Ret']
SG = result[result.Group == 'SG']['Ret']
SM = result[result.Group == 'SM']['Ret']
BV = result[result.Group == 'BV']['Ret']
BG = result[result.Group == 'BG']['Ret']
BM = result[result.Group == 'BM']['Ret']

SMB = ((SV + SM + SG) / 3) - ((BV + BM + BG) / 3)
SMB = pd.merge(Trdmnt, SMB, on='Trdmnt', how='left').rename(columns={'Ret':'SMB'})
SMB.SMB = SMB.SMB.shift(1)
SMB.SMB = SMB.SMB * 100



