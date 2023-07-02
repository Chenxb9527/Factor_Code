# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:26:14 2022

@author: Chenx
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

########################################################################
#                            分组函数                                  # 
########################################################################

# 计算加权重
def cpt_vw_ret(group, avg_name, weight_name):
    
    d = group[avg_name]
    w = group[weight_name]
    
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan    


# 计算股票分组
def get_stock_groups(data, sortname, groups_num):

    df = data.copy()
    labels = ['G0' + str(i) if i < 10 else 'G' + str(i) for i in range(1, groups_num + 1) ]
    try:
        groups = pd.DataFrame(pd.qcut(df[sortname], groups_num, labels=labels).astype(str)).rename(columns={sortname: 'Group'})
    except:
        groups = pd.DataFrame(pd.qcut(df[sortname].rank(method='first'), groups_num, labels=labels).astype(str)).rename(columns={sortname: 'Group'})


    return groups


# 获取单分组结果
def get_single_sort(data, sortname, TimeName, groups_num, weighted):
    
    df = data.copy()
    PortTag = df.groupby([TimeName]).apply(get_stock_groups, sortname, groups_num)
    df = pd.merge(df, PortTag['Group'], left_index=True, right_index=True)
    df['Weight'] = df[weighted] / df.groupby([TimeName, 'Group'])[weighted].transform('sum')  
    
    ret_name = 'F1_Ret'
    
    vwret = df.groupby([TimeName, 'Group']).apply(cpt_vw_ret, ret_name, 'Weight').to_frame().reset_index().rename(columns={0: 'Ret'})    
    vwret = vwret.set_index(TimeName) 
    # ewret = df.groupby([TimeName, 'Group'])[ret_name].mean().to_frame().reset_index().rename(columns={ret_name: 'Ret'})
    # ewret = ewret.set_index(TimeName)
    
    return vwret



########################################################################
#                            数据预处理函数                             # 
########################################################################


# MAD法
def extreme_feature_MAD(data, feature_name, num=3, p=1.4826):
    
    df = data.copy()
    median = df[feature_name].median()
    MAD = abs(df[feature_name] - median).median()
    df.loc[:, feature_name] = df.loc[:, feature_name].clip(lower=median-num*p*MAD, upper=median+num*p*MAD, axis=1)

    return df

    
# 标准化处理
def get_zscore(data, feature_name):

    df = data.copy()
    df[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    
    return df

# 市值、行业中性化（回归法）
def data_scale_neutral_size_ind(data, CAP_name, industry_name, feature_name):

    df = data.copy()
    df_ind = pd.get_dummies(df[industry_name], columns=[industry_name])
    df_ind[CAP_name] = df[CAP_name]
    X = np.array(df_ind)

    for name in feature_name:
        y = np.array(df[name])
        reg = sm.OLS(y, X).fit()
        df[name] = reg.resid

    return df


########################################################################
#                            评价指标函数                               # 
########################################################################


# 计算年化收益率
def get_annual_ret(df, m):
    
    annual_ret = (df + 1).prod() ** (m / len(df)) - 1
    
    return annual_ret


# 计算胜率
def get_win_ratio(df):
    
    win_ratio = pd.Series(dtype='float64')
    for i in df.columns:
        win_ratio_sig = len(df[df[i] > 0]) / len(df)
        win_ratio[i] = win_ratio_sig
    
    return win_ratio


# 计算净值
def get_NAV(df,cost):
    
    Strategy_NAV = (df - cost + 1.0).cumprod()
    
    return Strategy_NAV

# 计算年化波动率
def get_annual_vol(df, m):
    
    # 年化波动率 m = 12
    annual_vol = np.sqrt(m) * df.std()
    
    return annual_vol

# 计算最大回撤
def get_maxDrawdownRate(df):
    
    endDate = np.argmax((np.maximum.accumulate(df) - df) / np.maximum.accumulate(df))
    if endDate == 0:
        
        return 0
    else:
        startDate = np.argmax(df[:endDate])
        
        return (df[startDate] - df[endDate]) /df[startDate]
    

########################################################################
#                           IC函数                                     # 
########################################################################


# 获取因子rank值
def get_rank(df, TimeName, rank_list):

    rank_df = df.groupby([TimeName])[rank_list].rank()

    return rank_df


# 计算因子相关系数
def get_factor_corr(df, factorList, RetName, method='spearman'):

    corr_Res = df[factorList].corrwith(df[RetName], method=method)

    return corr_Res
   

# 计算因子IC
def cal_factor_ic(df, TimeName, factorList, method='spearman'):

    ic = df.groupby([TimeName]).apply(get_factor_corr, factorList, 'F1_Ret', method)

    return ic


