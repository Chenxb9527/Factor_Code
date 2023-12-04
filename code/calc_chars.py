# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime

# load the Dalyr data
def get_ST_PT_info():
    
    ST_PT = pd.read_csv('data/input/csmar_tables/TRD_Dalyr.csv')
    ST_PT.sort_values(by=['Trddt'], inplace=True, ignore_index=True)
    ST_PT['Trdmnt'] = list(ST_PT['Trddt'])
    ST_PT.Trdmnt = ST_PT.Trdmnt.apply(lambda x: x[0:4] + x[5:7])
    ST_PT.drop_duplicates(subset=['Trdmnt', 'Stkcd'], keep='last', inplace=True)
    ST_PT = ST_PT[['Stkcd', 'Trdmnt', 'Trdsta']]
    
    return ST_PT


###################################################################################
#                                 factor calculation                              # 
###################################################################################

class Ashare_calc_chars():
    
    def __init__(self):
        
        # ST/PT info the end of the Month
        info_ST = pd.read_csv('data/input/csmar_tables/ST_PT.csv')
        info_ST.Trdmnt = info_ST.Trdmnt.astype('str')
        info_ST = info_ST[info_ST.Trdmnt <= '202201']
        self.ST_PT = info_ST
        self.monthly_ret = self.get_data('TRD_Mnth', 'Mretwd')

    def drop_ST_PT(self, df):

        result = df.copy()
        info_ST = pd.read_csv('data/input/csmar_tables/ST_PT.csv')
        info_ST.Trdmnt = info_ST.Trdmnt.astype('str')
        info_ST = info_ST[info_ST.Trdmnt <= '202201']
        result = result.merge(info_ST, on=['Stkcd', 'Trdmnt'])
        result = result[result.Markettype.isin([1, 4, 16, 32])]
        result = result[result.Trdsta.isin([1, 4, 7, 10, 13])]
        result.drop(columns=['Trdsta'], inplace=True)
        
        return result
    
    def drop_FinInd(self, df, table):
        
        result = df.copy()
        Ind = pd.read_csv('data/input/csmar_tables/INDFI_INDTRAJECTORY.csv')
        Ind['year'] = Ind.EndDate.apply(lambda x: x[:4])
        Ind.rename(columns={'StockCode':'Stkcd'}, inplace=True)
        Ind = Ind[['Stkcd', 'IndustryCode', 'year']]
        
        if table == 'TRD_Mnth':
            result['year'] = result.Trdmnt.apply(lambda x: x[:4])     
        if table == 'TRD_Dalyr':
            result['year'] = result.Trddt.apply(lambda x: x[:4])
            
        result = pd.merge(result, Ind, on=['Stkcd', 'year'], how='left')
        # result['IndustryName'] = result.groupby(['Stkcd']).IndustryName.fillna(method='ffill')
        result.drop(index=result[(result.IndustryCode == 'J66') | 
                                 (result.IndustryCode == 'J67') | 
                                 (result.IndustryCode == 'J68') |
                                 (result.IndustryCode == 'J69') |
                                 (result.IndustryCode.isnull())].index, inplace=True)
        result.drop(columns=['year'], inplace=True)
        
        return result
    
    def drop_IPONxYear(self, df, table):
        
        result = df.copy()
        
        Listdt = pd.read_csv('data/input/csmar_tables/IPO_Cobasic.csv')
        Listdt.Listdt = Listdt.Listdt.apply(lambda x: x[:4] + x[5:7] + x[8:10])
        Listdt['Listdt_NxYear'] = pd.to_datetime(Listdt.Listdt)
        Listdt['Listdt_NxYear'] = Listdt['Listdt_NxYear'].apply(lambda x: x + datetime.timedelta(days=365))
        Listdt.Listdt_NxYear = Listdt.Listdt_NxYear.astype('str')
        Listdt['Listdt_NxYear'] = Listdt['Listdt_NxYear'].apply(lambda x: x[:4] + x[5:7])
        Listdt = Listdt[Listdt['Listdt'] <= '202201']
        Listdt = Listdt[['Stkcd','Listdt_NxYear']]
        
        if table == 'TRD_Mnth':
            result = pd.merge(result, Listdt, on=['Stkcd'], how='left')
            result = result[result.Trdmnt > result.Listdt_NxYear]
            result.drop(columns=['Listdt_NxYear'], inplace=True)
        if table == 'TRD_Dalyr':
            result['Trdmnt'] = result['Trddt'].apply(lambda x: x[:6])
            result = pd.merge(result, Listdt, on=['Stkcd'], how='left')
            result = result[result.Trdmnt > result.Listdt_NxYear]
            result.drop(columns=['Listdt_NxYear', 'Trdmnt'], inplace=True)
        
        return result
    
    def get_lag_data(self, raw_data, lag_rule):
        
        universe = self.monthly_ret.copy()
        data = universe.copy()
        data.iloc[:, :] = np.nan
        lag_data = raw_data.reset_index()
        lag_d = []
        if lag_rule == 'lag_rule1':
            for i in range(lag_data.shape[0]):
                if lag_data.loc[i]['Trdmnt'][4:] == '03':
                    lag_d.append(lag_data.loc[i]['Trdmnt'][0:4] + '04')
                if lag_data.loc[i]['Trdmnt'][4:] == '06':
                    lag_d.append(lag_data.loc[i]['Trdmnt'][0:4] + '08')
                if lag_data.loc[i]['Trdmnt'][4:] == '09':
                    lag_d.append(lag_data.loc[i]['Trdmnt'][0:4] + '10')
                if lag_data.loc[i]['Trdmnt'][4:] == '12':
                    newyear = str(int(lag_data.loc[i]['Trdmnt'][0:4]) + 1)
                    lag_d.append(newyear + '04')
                    
        if lag_rule == 'lag_rule2':
            for i in range(lag_data.shape[0]):
                if lag_data.loc[i]['Trdmnt'][4:] == '03':
                    lag_d.append(np.nan)
                if lag_data.loc[i]['Trdmnt'][4:] == '06':
                    lag_d.append(lag_data.loc[i]['Trdmnt'][0:4] + '09')
                if lag_data.loc[i]['Trdmnt'][4:] == '09':
                    lag_d.append(lag_data.loc[i]['Trdmnt'][0:4] + '11')
                if lag_data.loc[i]['Trdmnt'][4:] == '12':
                    newyear = str(int(lag_data.loc[i]['Trdmnt'][0:4]) + 1)
                    lag_d.append(newyear + '05')
        
        lag_data['Trdmnt'] = pd.DataFrame(lag_d)
        lag_data.dropna(subset=['Trdmnt'], inplace=True)
        lag_data.fillna(method='ffill', inplace=True)
        lag_data.drop_duplicates(subset=['Trdmnt'], keep='first', inplace=True)
        lag_data.set_index(['Trdmnt'], inplace=True)
        data.loc[:, :] = lag_data.loc[:, :]
        data.fillna(method='ffill', inplace=True)
        universe = np.isnan(universe)
        data[universe] = np.nan
        
        return data

    def get_data(self, table, field=None, fs_freq='q', lag=True, lag_rule=None):
        
        raw_data = pd.read_csv('data/input/csmar_tables/' + table + '.csv')
        
        if table == 'TRD_Mnth':
            raw_data.Trdmnt = raw_data.Trdmnt.apply(lambda x: x[0:4] + x[5:])
            raw_data = raw_data[raw_data.Trdmnt <= '202201']
            raw_data = self.drop_ST_PT(raw_data)
            raw_data = self.drop_IPONxYear(raw_data, table)
            raw_data = self.drop_FinInd(raw_data, table)
            data = raw_data.pivot(index='Trdmnt', columns='Stkcd', values=field)
            
        if table == 'TRD_Dalyr':
            raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
            raw_data = raw_data[raw_data.Trddt <= '20211130']
            raw_data = raw_data[raw_data.Markettype.isin([1, 4, 16, 32])]
            raw_data = raw_data[raw_data.Trdsta.isin([1, 4, 7, 10, 13])]
            raw_data = self.drop_IPONxYear(raw_data, table)
            raw_data = self.drop_FinInd(raw_data, table)
            data = raw_data.pivot(index='Trddt', columns='Stkcd', values=field)
        
        if table == 'TRD_Cndalym':
            raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
            raw_data = raw_data[(raw_data.Trddt >= '19930101') & (raw_data.Trddt <= '20211130')]
            raw_data = raw_data[raw_data.Markettype == 53]
            data = raw_data[['Trddt', field]]
                   
        if table == 'TRD_Nrrate':
            raw_data['Trddt'] = raw_data['Clsdt']
            raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
            raw_data = raw_data[(raw_data.Trddt >= '19930101') & (raw_data.Trddt <= '20211130')]
            data = raw_data[['Trddt', field]]
            data[field] = data[field].apply(lambda x: x / 100)
         
        if table == 'STK_MKT_THRFACDAY':
            raw_data = raw_data[raw_data.MarkettypeID == 'P9714']
            raw_data['Trddt'] = raw_data['TradingDate']
            raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
            raw_data = raw_data[(raw_data.Trddt >= '19930101') & (raw_data.Trddt <= '20211130')]
            data = raw_data[['Trddt', 'RiskPremium1', 'SMB1', 'HML1']]

        if table == 'STK_MKT_FIVEFACDAY':
            raw_data = raw_data[(raw_data.MarkettypeID == 'P9714') & (raw_data.Portfolios == 1)]
            raw_data['Trddt'] = raw_data['TradingDate']
            raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
            raw_data = raw_data[(raw_data.Trddt >= '19930101') & (raw_data.Trddt <= '20211130')]
            data = raw_data[['Trddt', 'RiskPremium1', 'SMB1', 'HML1','RMW1','CMA1']]
        
        if table == 'CH_3_fac_daily_update_20211231':
            raw_data.rf_dly = raw_data.rf_dly / 100
            raw_data.mktrf = raw_data.mktrf / 100
            raw_data.SMB = raw_data.SMB / 100
            raw_data.VMG = raw_data.VMG / 100
            raw_data.rename(columns={'date':'Trddt'}, inplace=True)
            data = raw_data
            
            
        if (table == 'FS_Combas') | (table == 'FS_Comins') | (table == 'FS_Comscfd') | (table == 'FS_Comscfi'):
            raw_data = pd.read_csv('data/input/csmar_tables/' + table + '.csv')
            raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
            raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])

            if fs_freq == 'y':
                raw_data.drop(index=raw_data[raw_data.type != '1231'].index, inplace=True)
                raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
                raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values=field)

            if fs_freq == 'q':
                raw_data.drop(index=raw_data[
                    (raw_data.type != '1231') & (raw_data.type != '0930') & (raw_data.type != '0630') & (
                            raw_data.type != '0331')].index, inplace=True)
                
                raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
                raw_data.rename(columns={'Accper':'Trdmnt'}, inplace=True)
                raw_data = raw_data.pivot(index='Trdmnt', columns='Stkcd', values=field)
                
                if lag == True:
                    data = self.get_lag_data(raw_data, lag_rule=lag_rule)
                
                if lag == False:
                    universe = self.monthly_ret.copy()
                    data = universe.copy()
                    data.iloc[:, :] = np.nan
                    data.loc[:, :] = raw_data.loc[:, :]
            
        return data
    

    ## 交易摩擦因子
    # Firm size
    # TIP Done perfectly
    def calc_size(self):
        
        size = self.get_data('TRD_Mnth', 'Msmvosd')
        # lncap = np.log(cap)
        
        return size
    
    # TIP Done perfectly
    # Industry-adjusted size，size_ia
    def calc_size_ia(self):
        
        size = self.calc_size()
        ind_table = self.get_data('TRD_Mnth', 'IndustryCode')
        indCode_list = []
        for i in range(ind_table.shape[1]):
            indCode_list = indCode_list + list(set(ind_table.iloc[:, i][~ind_table.iloc[:, i].isnull()].unique()))
        indCode_list = list(set(indCode_list))
        
        indAve = self.monthly_ret.copy()
        indAve.iloc[:, :] = np.nan
        for i in range(size.shape[0]):
            temp = ind_table.iloc[i, :]
            for j in indCode_list:
                pos = np.where(temp == j)[0]
                temp_size = np.array(size.iloc[i, pos])
                temp_indAve = np.nanmean(temp_size)
                indAve.iloc[i, pos] = temp_indAve
        size_ia = size - indAve

        return size_ia
    
    # TIP Done perfectly
    # market beta, beta    
    def calc_beta(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        rf = np.array(mkt_and_rf.iloc[:, -1])
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan

        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp.mkt_excess)
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    beta_temp = ols.params[1]
                    result.loc[k, i] = beta_temp
                else:
                    result.loc[k, i] = np.nan
             
        return result
    
    # TIP Done perfectly
    # square of market beta, betasq
    def calc_betasq(self):
        
        beta = self.calc_beta()
        betasq = np.square(beta)
        
        return betasq
    
    # TIP Done perfectly
    # The Dimson Beta (betaDM)
    def calc_betad(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)

        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(rf, mkt_ret, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess']]
            ret_temp.dropna(inplace=True)
            ret_temp['mkt_excess_after'] = ret_temp.mkt_excess.shift(1)
            ret_temp['mkt_excess_before'] = ret_temp.mkt_excess.shift(-1)
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp[['mkt_excess', 'mkt_excess_after', 'mkt_excess_before']])
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    beta_temp = ols.params[1] + ols.params[2] + ols.params[3]
                    result.loc[k, i] = beta_temp
                else:
                    result.loc[k, i] = np.nan
             
        return result
    
    # idiosyncratic volatility, idvol
    def calc_idvol(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)

        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp.mkt_excess)
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = np.sqrt(ols.mse_resid * ols.df_resid / (ols.df_model + 1 + ols.df_resid))
                else:
                    result.loc[k, i] = np.nan
             
        return result
    
    def calc_idvol_ff3(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff3 = self.get_data('STK_MKT_THRFACDAY')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        daily_ret = pd.merge(daily_ret, ff3, on='Trddt', how='left')
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'RiskPremium1', 'SMB1', 'HML1']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j]
                if len(temp) >= 10:
                    y = np.array(temp[i])
                    X = np.array(temp[['RiskPremium1', 'SMB1', 'HML1']])
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = ols.resid.std()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    def calc_idvol_ff5(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff5 = self.get_data('STK_MKT_FIVEFACDAY')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        daily_ret = pd.merge(daily_ret, ff5, on='Trddt', how='left')
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count','RiskPremium1', 'SMB1', 'HML1', 'RMW1', 'CMA1']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j]
                if len(temp) >= 10:
                    y = np.array(temp[i])
                    X = np.array(temp[['RiskPremium1', 'SMB1', 'HML1', 'RMW1', 'CMA1']])
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = ols.resid.std()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    def calc_idvol_ch3(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ch3 = self.get_data('CH_3_fac_daily_update_20211231')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        # rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        # mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        # mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        # daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        # daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        ch3.Trddt = ch3.Trddt.astype('str')
        daily_ret = pd.merge(daily_ret, ch3, on='Trddt', how='left')
        rf = np.array(daily_ret.rf_dly)
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mktrf', 'SMB', 'VMG']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j]
                if len(temp) >= 10:
                    y = np.array(temp[i])
                    X = np.array(temp[['mktrf', 'SMB', 'VMG']])
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = ols.resid.std()
                else:
                    result.loc[k, i] = np.nan
        
        return result

    # TIP Done perfectly
    # Total Volatility (tv)
    def calc_vol(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j][i]
                if len(temp) >= 10:
                    result.loc[k, i] = temp.std(ddof=0)
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # idiosyncratic skewness, idskew
    def calc_idskew(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp.mkt_excess)
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    temp_iv = np.sqrt(ols.mse_resid * ols.df_resid / (ols.df_model + 1 + ols.df_resid))
                    temp_res = ols.predict() - y
                    temp_num = ols.df_model + 1 + ols.df_resid
                    result.loc[k, i] = np.sum(np.power(temp_res, 3)) / (temp_num * np.power(temp_iv, 3))
                else:
                    result.loc[k, i] = np.nan
             
        return result

    # TIP Done perfectly
    # total skewness, skew
    def calc_skew(self):
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count']]
            ret_temp[i] = ret_temp[i].shift(1)
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)][i]
                if len(temp) >= 120:
                    result.loc[k, i] = temp.skew()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # coskewness, coskew
    def calc_coskew(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        daily_ret['mkt_excess^2'] = daily_ret.mkt_excess ** 2
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess', 'mkt_excess^2']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp[['mkt_excess', 'mkt_excess^2']])
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = ols.params[-1]
                else:
                    result.loc[k, i] = np.nan
             
        return result
    
    # TIP Done perfectly
    # turnover, turn
    def calc_turn(self):
        
        shares = self.get_data('TRD_Dalyr', 'Dnshrtrd')
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        mv = mv * 1000
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv / price
        turnover = shares / number
        stkcd_list = list(turnover.columns)
        
        turnover.reset_index(inplace=True)
        turnover['Trdmnt'] = list(turnover.Trddt)
        turnover.Trdmnt = turnover.Trdmnt.apply(lambda x: x[0:6])

        months_count = turnover[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        turnover = pd.merge(turnover, months_count, on='Trdmnt', how='left')
                           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            turnover_temp = turnover[[i, 'months_count']]
            turnover_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = turnover_temp[(turnover_temp['months_count'] <= j ) & (turnover_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    result.loc[k, i] = temp[i].mean()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # volatility of turnover, std_turn
    def calc_std_turn(self):
        
        shares = self.get_data('TRD_Dalyr', 'Dnshrtrd') 
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        mv = mv * 1000
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv / price
        turnover = shares / number
        stkcd_list = list(turnover.columns)
        
        turnover.reset_index(inplace=True)
        turnover['Trdmnt'] = list(turnover.Trddt)
        turnover.Trdmnt = turnover.Trdmnt.apply(lambda x: x[0:6])

        months_count = turnover[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        turnover = pd.merge(turnover, months_count, on='Trdmnt', how='left')
        
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            turnover_temp = turnover[[i, 'months_count']]
            turnover_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = turnover_temp[turnover_temp['months_count'] == j]
                if len(temp) >= 10:
                    result.loc[k, i] = temp[i].std()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # volume in dollar, volumed
    def calc_volumed(self):
        
        volume = self.get_data('TRD_Dalyr', 'Dnvaltrd')
        stkcd_list = list(volume.columns)
        
        volume.reset_index(inplace=True)
        volume['Trdmnt'] = list(volume.Trddt)
        volume.Trdmnt = volume.Trdmnt.apply(lambda x: x[0:6])

        months_count = volume[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        volume = pd.merge(volume, months_count, on='Trdmnt', how='left')
                           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            volumed_temp = volume[[i, 'months_count']]
            volumed_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = volumed_temp[(volumed_temp['months_count'] <= j ) & (volumed_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    result.loc[k, i] = temp[i].mean()
                else:
                    result.loc[k, i] = np.nan
             
        return result
    
    # TIP Done perfectly
    # volatility of volume in dollar, std_vol
    def calc_std_vol(self):
        
        volume = self.get_data('TRD_Dalyr', 'Dnvaltrd')
        stkcd_list = list(volume.columns)
        
        volume.reset_index(inplace=True)
        volume['Trdmnt'] = list(volume.Trddt)
        volume.Trdmnt = volume.Trdmnt.apply(lambda x: x[0:6])

        months_count = volume[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        volume = pd.merge(volume, months_count, on='Trdmnt', how='left')
                           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            volumed_temp = volume[[i, 'months_count']]
            volumed_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = volumed_temp[volumed_temp['months_count'] == j]
                if len(temp) >= 10:
                    result.loc[k, i] = temp[i].std()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # illiquidity, illq
    def calc_illq(self):
        
        volume = self.get_data('TRD_Dalyr', 'Dnvaltrd')
        daily_ret = abs(self.get_data('TRD_Dalyr', 'Dretwd'))
        Ami = daily_ret / volume
        stkcd_list = list(Ami.columns)
        
        Ami.reset_index(inplace=True)
        Ami['Trdmnt'] = list(Ami.Trddt)
        Ami.Trdmnt = Ami.Trdmnt.apply(lambda x: x[0:6])

        months_count = Ami[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        Ami = pd.merge(Ami, months_count, on='Trdmnt', how='left')
                           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            Ami_temp = Ami[[i, 'months_count']]
            Ami_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = Ami_temp[(Ami_temp['months_count'] <= j ) & (Ami_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    result.loc[k, i] = temp[i].mean() * (10**6)
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # TIP Done perfectly
    # maximum daily return, retnmat
    def calc_retmax(self):
                    
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j][i]
                if len(temp) >= 10:
                    result.loc[k, i] = temp.max()
                else:
                    result.loc[k, i] = np.nan
        
        return result
    
    # ???
    # LM 标准换手率
    def calc_LM(self, trading_day_num=21):
        
        universe = self.get_data('TRD_Mnth', 'Mretwd').copy()
        universe = np.isnan(universe)
        
        volume = self.get_data('TRD_Dalyr', 'Dnshrtrd') 
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        mv = mv * 1000
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv / price
        turnover_all = volume / number
        turnover_all['Trdmnt'] = list(turnover_all.index)
        turnover_all.Trdmnt = turnover_all.Trdmnt.apply(lambda x: x[0:6])
        
        turnover = universe.copy()
        turnover.iloc[:, :] = np.nan
        for i in list(turnover_all.Trdmnt.unique()):
            temp = turnover_all[turnover_all.Trdmnt == i]
            temp.set_index('Trdmnt', inplace=True)
            temp.fillna(0, inplace=True)
            temp = temp.cumsum(axis=0)
            temp = temp.reset_index().drop_duplicates(subset=['Trdmnt'], keep='last')
            temp = temp.replace(0, np.nan)
            temp.set_index('Trdmnt', inplace=True)
            turnover.loc[i, :] = temp.loc[i, :]
        
        NoTD = self.get_data('TRD_Mnth', 'Ndaytrd')
        trading_days_num = turnover_all[['Trdmnt']].value_counts().sort_index()
        trading_days = universe.copy()
        trading_days.iloc[:, :] = np.nan
        for i in list(turnover_all.Trdmnt.unique()):
            trading_days.loc[i] = trading_days_num.loc[i].values[0]
        Nozd = trading_days - NoTD
        
        LM = (Nozd + 1 / (turnover)) * (21 / NoTD)   
        
        return LM
    
    # TIP Done perfectly
    # changes in shares outstanding, sharechg 
    def calc_sharechg(self):
        
        mv = self.get_data('TRD_Mnth', 'Msmvosd')
        mv = mv * 1000
        price = self.get_data('TRD_Mnth', 'Mclsprc')
        number = mv / price
        sharechg = number / number.shift(11) - 1
        
        return sharechg
    
    # TIP Done perfectly
    # Firm Age
    def calc_age(self):
        
        listdate = pd.read_csv('data/input/csmar_tables/IPO_Cobasic.csv')
        listdate.set_index('Stkcd', inplace=True)
        result = self.get_data('TRD_Mnth', 'Mretwd')
        result.iloc[:, :] = np.nan
        
        for i in listdate.index:
            if i not in result.columns:
                listdate.drop(index=listdate[listdate.index == i].index,inplace=True)
                
        for i in result.index:
            for j in list(result.columns):
                months = (int(i[:4])-int(listdate.loc[j].values[0][:4]))*12 + int(i[4:6])-int(listdate.loc[j].values[0][5:7]) + 1
                if months > 12:
                    years = int(months / 12)
                    result.loc[i, j] = years
                else:
                    result.loc[i, j] = np.nan
                if months == 12:
                    result.loc[i, j] = np.nan

        return result
   
    
    ## 动量因子
    
    # momentum，mom6 mom12
    # TIP Done perfectly 
    # mom12 end of month t-12 to end of month t-1
    # mom6 end of month t-6 to end of month t-1
    # mom36 end of month t-36 to end of month t-13
    
    def calc_mom(self, start, end):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        timeline = pd.DataFrame(pd.date_range('1993-01-01','2021-11-30',freq='D'))[0].dt.strftime('%Y%m%d')
        df = pd.DataFrame(timeline).rename(columns={0: 'Trddt'})
        ret = pd.merge(df, daily_ret, on='Trddt', how='left')
        ret['Trdmnt'] = ret.Trddt
        ret['Trdmnt'] = ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        ret = pd.merge(ret, months_count, on='Trdmnt', how='left')
        ret.iloc[:, 1:-2] = ret.iloc[:, 1:-2].shift(1)
        
        mom = self.get_data('TRD_Mnth', 'Mretwd').copy()
        mom.iloc[:, :] = np.nan

        for i in stkcd_list:
            ret_temp = ret[[i, 'months_count']]
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j - start) & (ret_temp['months_count'] >= j - end)][i]
                
                if len(temp[~np.isnan(temp)]) > 0:             
                    temp.iloc[0] = 0
                    ret_cum = (temp + 1).cumprod()
                    mom.loc[k, i] = ret_cum[~np.isnan(ret_cum)].iloc[-1]
                else:
                    mom.loc[k, i] = np.nan
             
        return mom

    # Momentum Change (mchg)
    def calc_mchg(self):

        m1 = self.calc_mom(1, 6)
        m2 = self.calc_mom(7, 11)
        result = m1 - m2
        
        return result
    
    # idiosyncratic momentum, imom
    def calc_imom(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)

        daily_ret.reset_index(inplace=True)
        daily_ret['Trdmnt'] = list(daily_ret.Trddt)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = daily_ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        daily_ret = pd.merge(daily_ret, months_count, on='Trdmnt', how='left')
        
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        mkt_ret = self.get_data('TRD_Cndalym', 'Cdretwdeq')
        mkt_and_rf = pd.merge(mkt_ret, rf, on=['Trddt'], how='left')
        daily_ret = pd.merge(daily_ret, mkt_and_rf, on='Trddt', how='left')
        daily_ret['mkt_excess'] = daily_ret.Cdretwdeq - daily_ret.Nrrdaydt
        rf = np.array(mkt_and_rf.iloc[:, -1])
        
        for i in stkcd_list:
           daily_ret.loc[:, i] = daily_ret.loc[:, i] - rf
           
        result = self.get_data('TRD_Mnth', 'Mretwd').copy()
        result.iloc[:, :] = np.nan
        imom = result.copy()
        
        for i in stkcd_list:
            ret_temp = daily_ret[[i, 'months_count', 'mkt_excess']]
            ret_temp.dropna(inplace=True)
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[(ret_temp['months_count'] <= j ) & (ret_temp['months_count'] >= j - 11)]
                if len(temp) >= 120:
                    y = np.array(temp[i])
                    X = np.array(temp.mkt_excess)
                    X = sm.add_constant(X)
                    ols = sm.OLS(y, X).fit()
                    result.loc[k, i] = np.sqrt(ols.mse_resid * ols.df_resid / (ols.df_model + 1 + ols.df_resid))
                else:
                    result.loc[k, i] = np.nan

    
    # short-term reversal, lagretn
    def calc_lagretn(self):
        
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        stkcd_list = list(daily_ret.columns)
        
        timeline = pd.DataFrame(pd.date_range('1993-01-01','2021-11-30',freq='D'))[0].dt.strftime('%Y%m%d')
        df = pd.DataFrame(timeline).rename(columns={0: 'Trddt'})
        ret = pd.merge(df, daily_ret, on='Trddt', how='left')
        ret['Trdmnt'] = list(ret.Trddt)
        ret['Trdmnt'] = ret.Trdmnt.apply(lambda x: x[0:6])

        months_count = ret[['Trdmnt']]
        months_count.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        months_count.reset_index(drop=True, inplace=True)
        months_count['months_count'] = months_count.index + 1
        ret = pd.merge(ret, months_count, on='Trdmnt', how='left')
        ret.iloc[:, 1:-2] = ret.iloc[:, 1:-2].shift(1)
        
        lagretn = self.get_data('TRD_Mnth', 'Mretwd').copy()
        lagretn.iloc[:, :] = np.nan

        for i in stkcd_list:
            ret_temp = ret[[i, 'months_count']]
            for k, j in zip(months_count['Trdmnt'], months_count['months_count']):
                temp = ret_temp[ret_temp['months_count'] == j][i]
                temp.iloc[0] = 0
                ret_cum = (temp + 1).cumprod()
                lagretn.loc[k, i] = ret_cum[~np.isnan(ret_cum)].iloc[-1]
        
        universe = self.monthly_ret().copy()
        universe = np.isnan(universe)
        lagretn[universe] = np.nan
        
        return lagretn
    
    
    ## 价值因子
    # book-to-market ratio, BM
    def calc_bm(self):
        
        book_value = self.get_data('FS_Combas', 'A003000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        msmvttl = self.get_data('TRD_Mnth', 'Msmvttl')
        # market_equity = market_equity * 1000
        # min_int = self.get_data('FS_Combas', 'A003200000', lag=False)
        # min_int.fillna(0, inplace=True)
        # bm = (book_value-min_int) / market_equity
        bm = book_value / market_equity 
        bm.dropna(axis=0, how='all', inplace=True)
        bm = self.get_lag_data(bm, lag_rule='lag_rule1')
        
        return bm
    
    # Industry adjusted book-to-market ratio，BM_ia
    def calc_bm_ia(self):
        
        bm = self.calc_bm()
        ind_table = self.get_data('TRD_Mnth', 'IndustryCode')
        indCode_list = []
        for i in range(ind_table.shape[1]):
            indCode_list = indCode_list + list(set(ind_table.iloc[:, i][~ind_table.iloc[:, i].isnull()].unique()))
        indCode_list = list(set(indCode_list))
        
        indAve = self.monthly_ret.copy()
        indAve.iloc[:, :] = np.nan
        for i in range(bm.shape[0]):
            temp = ind_table.iloc[i, :]
            for j in indCode_list:
                pos = np.where(temp == j)[0]
                temp_bm = np.array(bm.iloc[i, pos])
                temp_indAve = np.nanmean(temp_bm)
                indAve.iloc[i, pos] = temp_indAve
        bm_ia = bm - indAve

        return bm_ia
    
    # TIP Done perfectly
    # asset-to-market ratio, AM
    def calc_am(self):
        
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        am = ttl_assets / market_equity
        am.dropna(axis=0, how='all', inplace=True)
        am = self.get_lag_data(am, lag_rule='lag_rule1')
        
        return am
    
    # TIP Done perfectly
    # liabilities-to-market ratio, LEV
    def calc_lev(self):
        
        ttl_liab = self.get_data('FS_Combas', 'A002000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        lev = ttl_liab / market_equity
        lev.dropna(axis=0, how='all', inplace=True)
        lev = self.get_lag_data(lev, lag_rule='lag_rule1')

        return lev
    
    # TIP Done perfectly
    # earnings-to-price ratio, EP
    def calc_ep(self):
        
        earings = self.get_data('FS_Comins', 'B002000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')        
        ep = earings / market_equity
        ep.dropna(axis=0, how='all', inplace=True)
        ep = self.get_lag_data(ep, lag_rule='lag_rule1')

        return ep
    
    # TIP Done perfectly
    # cfp        
    def calc_cfp(self):
        
        cash_flows = self.get_data('FS_Comscfd', 'C005000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        cfp = cash_flows / market_equity
        cfp.dropna(axis=0, how='all', inplace=True)
        cfp = self.get_lag_data(cfp, lag_rule='lag_rule1')

        return cfp
    
    # cfp_ia
    def calc_cfp_ia(self):
        
        cfp = self.calc_cfp()
        ind_table = self.get_data('TRD_Mnth', 'IndustryCode')
        indCode_list = []
        for i in range(ind_table.shape[1]):
            indCode_list = indCode_list + list(set(ind_table.iloc[:, i][~ind_table.iloc[:, i].isnull()].unique()))
        indCode_list = list(set(indCode_list))
        
        indAve = self.monthly_ret.copy()
        indAve.iloc[:, :] = np.nan
        for i in range(cfp.shape[0]):
            temp = ind_table.iloc[i, :]
            for j in indCode_list:
                pos = np.where(temp == j)[0]
                temp_cfp = np.array(cfp.iloc[i, pos])
                temp_indAve = np.nanmean(temp_cfp)
                indAve.iloc[i, pos] = temp_indAve
        cfp_ia = cfp - indAve
        
        return cfp_ia
    
    # TIP Done perfectly
    # operating cash-flow-to-price ratio, OCFP
    def calc_ocfp(self):
        
        op_cash_flow = self.get_data('FS_Comscfd', 'C001000000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        ocfpq = op_cash_flow / market_equity
        ocfpq.dropna(axis=0, how='all', inplace=True)
        ocfpq = self.get_lag_data(ocfpq, lag_rule='lag_rule1')

        return ocfpq
    
    # TIP Done perfectly
    # dividend-to-price ratio, DP
    def calc_dp(self):
        
        dividend = self.get_data('FS_Combas', 'A002115000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        dp = dividend / market_equity
        dp.dropna(axis=0, how='all', inplace=True)
        dp = self.get_lag_data(dp, lag_rule='lag_rule1')

        return dp
    
    # TIP Done perfectly
    # sales-to-price ratio, SP
    def calc_sp(self):
        
        op_re = self.get_data('FS_Comins', 'B001100000', lag=False)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        sp = op_re / market_equity
        sp.dropna(axis=0, how='all', inplace=True)
        sp = self.get_lag_data(sp, lag_rule='lag_rule1')

        return sp
    
    
    ## 成长因子
    # TIP Done perfectly
    # Asset growth ratio, AG
    def calc_ag(self):
        
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule1')
        ag = (ttl_assets - ttl_assets.shift(12)) / ttl_assets.shift(12)
        
        return ag
    
    # TIP Done perfectly
    # liabilities growth, LG
    def calc_lg(self):
        
        ttl_liab = self.get_data('FS_Combas', 'A002000000', lag_rule='lag_rule1')
        lg = (ttl_liab - ttl_liab.shift(12)) / ttl_liab.shift(12)

        return lg
    
    # TIP Done perfectly
    # book market value growth, BVEG
    def calc_bveg(self):
        
        bmv = self.get_data('FS_Combas', 'A003000000', lag_rule='lag_rule1')
        bveg = (bmv - bmv.shift(12)) / bmv.shift(12)
# 
        return bveg        
    
    # TIP Done perfectly
    # Sales growth，SG
    def calc_sg(self):
        
        op_re = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule1')
        sg = (op_re - op_re.shift(12)) / op_re.shift(12)
        
        return sg
    
    # TIP Done perfectly
    # Profit margin growth，PMG
    def calc_pmg(self):
        
        pm = self.get_data('FS_Comins', 'B001300000', lag_rule='lag_rule1')
        pmg = (pm - pm.shift(12)) / pm.shift(12)
        
        return pmg
    
    # TIP Done perfectly
    # inventory growth, INVG
    def calc_invg(self):
        
        inv = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule1')
        universe = np.isnan(inv)
        inv.replace({0:np.nan}, inplace=True)
        inv.fillna(method='ffill', inplace=True)
        inv[universe] = np.nan
        invg = (inv - inv.shift(12)) / inv.shift(12)
                
        return invg
    
    # TIP Done perfectly
    # inventory change, INVchg
    def calc_invchg(self):
        
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule1')
        ave_assets = (ttl_assets + ttl_assets.shift(12)) / 2
        inv = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule1')
        universe = np.isnan(inv)
        inv.replace({0:np.nan}, inplace=True)
        inv.fillna(method='ffill', inplace=True)
        inv[universe] = np.nan
        invchg = (inv - inv.shift(12)) / ave_assets
        
        return invchg   
    
    # TIP Done perfectly
    # sales growth minus inventory growth, SgINVg
    def calc_sginvg(self):
        
        sg = self.calc_sg()
        invg = self.calc_invg()
        sginvg = sg - invg
        
        return sginvg
    
    # TIP Done perfectly
    # tax growth, TAXchg
    def calc_taxchg(self):
        
        tax = self.get_data('FS_Combas', 'A002113000', lag_rule='lag_rule1')
        universe = np.isnan(tax)
        tax.replace({0:np.nan}, inplace=True)
        tax.fillna(method='ffill', inplace=True)
        tax[universe] = np.nan
        taxchg = (tax - tax.shift(12)) / tax.shift(12)

        return taxchg
    
    # TIP Done perfectly
    # Accruals Component  ACC
    def calc_acc(self):
        
        income = self.get_data('FS_Comins', 'B001000000', lag_rule='lag_rule2')
        op_cash = self.get_data('FS_Comscfd', 'C001000000', lag_rule='lag_rule2')
        accruals = income - op_cash
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2')
        ave_assets = (ttl_assets + ttl_assets.shift(12)) / 2
        acc = accruals / ave_assets
        
        return acc
            
    # TIP Done perfectly
    # Absolute value of acc absacc
    def calc_absacc(self):
        
        absacc = abs(self.calc_acc())
        
        return absacc
    
    # Accrual volatility，stdacc
    def calc_stdacc(self):
        
        acc = self.calc_acc()
        stdacc = acc.rolling(48).std()
        
        return stdacc
    
    # TIP Done perfectly
    # Percent accruals，ACCP
    def calc_accp(self):
        
        income = self.get_data('FS_Comins', 'B001000000', lag_rule='lag_rule2')
        op_cash = self.get_data('FS_Comscfd', 'C001000000', lag_rule='lag_rule2')
        net_income = self.get_data('FS_Comins', 'B002000000', lag_rule='lag_rule2')
        accp = (income - op_cash) / net_income
        
        return accp
    
    # Corporate investment，cinvest
    def calc_cinvest(self):
        
        fixed_assets = self.get_lagdata('FS_Combas', 'A001212000', 'cinvest')
        op_re = self.get_lagdata('FS_Comins', 'B001101000', 'cinvest')
        fixed_assets = fixed_assets.rolling(9).mean()
        op_re = op_re.rolling(9).mean()
        ce = fixed_assets / op_re
        ce = ce.shift(3)
        
        return ce
    
    # TIP Done perfectly
    # Depreciation / PP&E，depr
    def calc_depr(self):
        
        fixed_assets = self.get_data('FS_Combas', 'A001212000', lag_rule='lag_rule1')
        dep = self.get_data('FS_Comscfi', 'D000103000', lag_rule='lag_rule2')
        depr = dep / fixed_assets
        
        return depr
    
    # TIP Done perfectly
    # change in depreciation，pchdepr
    def calc_pchdepr(self):
        
        dep = self.get_data('FS_Comscfi', 'D000103000', lag=False)
        dep.dropna(axis=0, how='all', inplace=True)
        pchdepr = dep.pct_change()
        pchdepr.replace({0:np.nan}, inplace=True)
        pchdepr += 1
        pchdepr = self.get_lag_data(pchdepr, lag_rule='lag_rule2')
        
        return pchdepr
    
    # Change in shareholders’ equity，egr
    def calc_egr(self):
        
        se = self.get_data('FS_Combas', 'A003000000', lag=False)
        egr = se / se.shift(12)
        
        return egr
    
    # TIP Done perfectly
    # Percent change in capital expenditures，grCAPX
    def calc_grcapx(self):
        
        capx = self.get_data('FS_Comscfd','C002006000', lag=False)
        capx.dropna(axis=0, how='all', inplace=True)
        for i in capx.index:
            if i[4:] != '12':
                capx.drop(index=i, inplace=True)
        grcapx = capx.pct_change()
        grcapx.replace({0:np.nan}, inplace=True)
        grcapx += 1
        grcapx = self.get_lag_data(grcapx, lag_rule='lag_rule2')

        return grcapx
    
    # TIP Done perfectly
    # Industry adjusted % change in capital expenditures，pchcapx_ia
    def calc_pchcapx_ia(self):
        
        grcapx = self.calc_grcapx()
        ind_table = self.get_data('TRD_Mnth', 'IndustryCode')
        indCode_list = []
        for i in range(ind_table.shape[1]):
            indCode_list = indCode_list + list(set(ind_table.iloc[:, i][~ind_table.iloc[:, i].isnull()].unique()))
        indCode_list = list(set(indCode_list))
        
        indAve = self.monthly_ret.copy()
        indAve.iloc[:, :] = np.nan
        for i in range(grcapx.shape[0]):
            temp = ind_table.iloc[i, :]
            for j in indCode_list:
                pos = np.where(temp == j)[0]
                temp_grcapx = np.array(grcapx.iloc[i, pos])
                temp_indAve = np.nanmean(temp_grcapx)
                indAve.iloc[i, pos] = temp_indAve
        pchcapx_ia = grcapx - indAve

        return pchcapx_ia
    
    # TIP Done perfectly
    # Growth in long-term net operating assets，grltnoa
    def calc_grltnoa(self):
        
        noa = self.calc_noa()
        universe = np.isnan(noa)
        grltnoa = noa.pct_change()
        grltnoa.replace({0:np.nan}, inplace=True)
        grltnoa.fillna(method='ffill', inplace=True)
        grltnoa[universe] = np.nan
        grltnoa += 1

        return noa
    
    # TIP Done perfectly
    # Capital expenditures and inventory，invest
    def calc_invest(self):
        
        grcapx = self.calc_grcapx()
        inv = self.get_data('FS_Combas', 'A001123000', lag=False)
        inv.dropna(axis=0, how='all', inplace=True)
        for i in inv.index:
            if i[4:] != '12':
                inv.drop(index=i, inplace=True)
        invchg = inv.pct_change()
        invchg.replace({0:np.nan}, inplace=True)
        invchg += 1
        invchg = self.get_lag_data(invchg, lag_rule='lag_rule2')

        invest = grcapx + invchg
        
        return invest
        
    # % change in sales - % change in inventory，pchsale_pchinvt
    def calc_pchsale_pchinvt(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        sale.dropna(axis=0, how='all', inplace=True)
        pchsale = (sale - sale.shift(12)) / sale.shift(12)
        
        pchsale = sale.pct_change()
        pchsale.replace({0:np.nan}, inplace=True)
        pchsale += 1
        pchsale = self.get_lag_data(pchsale, lag_rule='lag_rule2')

        inv = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule2')
        inv.dropna(axis=0, how='all', inplace=True)
        pchinvt = (inv - inv.shift(12)) / inv.shift(12)

        pchinvt = inv.pct_change()
        pchinvt.replace({0:np.nan}, inplace=True)
        pchinvt += 1
        pchinvt = self.get_lag_data(pchinvt, lag_rule='lag_rule1')

        pchsale_pchinvt = pchsale - pchinvt
        
        return pchsale_pchinvt
    
    # change in sales - % change in A/R"，pchsale_pchrect
    def calc_pchsale_pchrect(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag=False)
        sale.dropna(axis=0, how='all', inplace=True)
        pchsale = sale.pct_change()
        pchsale.replace({0:np.nan}, inplace=True)
        pchsale += 1
        pchsale = self.get_lag_data(pchsale, lag_rule='lag_rule2')
        
        rect  = self.get_data('FS_Combas', 'A001111000', lag=False)
        rect.dropna(axis=0, how='all', inplace=True)
        pchrect = rect.pct_change()
        pchrect.replace({0:np.nan}, inplace=True)
        pchrect += 1
        pchrect = self.get_lag_data(pchrect, lag_rule='lag_rule2')

        pchsale_pchrect = pchsale - pchrect
        
        return pchsale_pchrect
    
    # % change in sales - % change in SG&A，pchsale_pchxsga
    
    # TIP Done perfectly
    # Real estate holdings，realestate
    def calc_realestate(self):
        
        realstate = self.get_data('FS_Combas', 'A001212000', lag_rule='lag_rule2')
        
        return realstate
    
    # TIP Done perfectly
    # Sales growth，sgr
    def calc_sgr(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        sgr = sale / sale.shift(12)
        
        return sgr
    
    # TIP Done perfectly
    # Net operating assets, noa
    def calc_noa(self):
        
        cash = self.get_data('FS_Combas', 'A001101000', lag_rule='lag_rule2')
        short_inv = self.get_data('FS_Combas', 'A001109000', lag_rule='lag_rule2')
        short_debt = self.get_data('FS_Combas', 'A002101000', lag_rule='lag_rule2')
        long_debt = self.get_data('FS_Combas', 'A002201000', lag_rule='lag_rule2')
        min_in = self.get_data('FS_Combas', 'A003200000', lag_rule='lag_rule2')
        ttl_holders_com = self.get_data('FS_Combas', 'A003100000', lag_rule='lag_rule2')
        noa = short_debt + long_debt + min_in + ttl_holders_com - cash - short_inv 
        
        return noa

    # hire
    def calc_hire(self):
        
        pass
    
    # TIP Done perfectly
    # research and development, RD
    def calc_rd(self):
        
        management_fee = self.get_data('FS_Comins', 'B001210000', lag_rule='lag_rule1')
        
        return management_fee
    
    
    # TIP Done perfectly
    # R&D to market capitalization，rd_mve
    def calc_rd_mve(self):
        
        management_fee = self.get_data('FS_Comins', 'B001210000', lag_rule='lag_rule1')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')        
        rd_mve = management_fee / market_equity

        return rd_mve
    
    # TIP Done perfectly
    # R&D to sale, RDsale
    def calc_rdsale(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule1')
        management_fee = self.get_data('FS_Comins', 'B001210000', lag_rule='lag_rule1')
        rdsale = management_fee / sale
        
        return rdsale
    
    
    ## 盈利因子
    # TIP Done perfectly
    # ROE
    def calc_roe(self):
        
        net_income = self.get_data('FS_Comins', 'B002000000', lag_rule='lag_rule2')
        oq_lag_bv = self.get_data('FS_Combas', 'A003000000', lag_rule='lag_rule2').shift(12)
        prefer = self.get_data('FS_Combas', 'A003112101', lag_rule='lag_rule2').shift(12)
        prefer.fillna(0, inplace=True)
        roe = net_income / (oq_lag_bv - prefer)
        
        return roe
    
    # TIP Done perfectly
    # ROA
    def calc_roa(self):
        
        net_income = self.get_data('FS_Comins', 'B002000000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2').shift(12)
        roa = net_income / ttl_assets
        
        return roa
    
    # TIP Done perfectly
    # CT
    def calc_ct(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2').shift(12)
        ct = sale / ttl_assets
        
        return ct
    
    # TIP Done perfectly
    # PA
    def calc_pa(self):
        
        income = self.get_data('FS_Comins', 'B001000000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2').shift(12)
        pa = income / ttl_assets
        
        return pa
    
    # TIP Done perfectly
    # cashspr 
    def calc_cashpr(self):
        
        longterm_liab = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A002206000', lag_rule='lag_rule2')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        # market_equity = market_equity * 1000
        cash = self.get_data('FS_Combas', 'A001101000', lag_rule='lag_rule2')
        cash.replace({0:np.nan}, inplace=True)
        cashpr = (market_equity + longterm_liab - ttl_assets) / cash
        
        return cashpr
    
    # TIP Done perfectly
    # cash
    def calc_cash(self):
        
        cash_equ = self.get_data('FS_Comscfd', 'C006000000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2')
        cash = cash_equ / ttl_assets
        
        return cash
    
    # Operating profit rate, operprof
    def calc_operprof(self):
        
        # oper = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        # other_oper = self.get_data('FS_Comins', 'B0f1105000', lag_rule='lag_rule2')
        # sale_cost = self.get_data('FS_Comins', 'B001209000', lag_rule='lag_rule2')
        # fin_cost = self.get_data('FS_Comins', 'B001211000', lag_rule='lag_rule2')
        # management_fee = self.get_data('FS_Comins', 'B001210000', lag_rule='lag_rule2')
        # equity = self.get_data('FS_Combas', 'A003101000', lag_rule='lag_rule2')
        # operprof = (oper + other_oper - sale_cost - fin_cost - management_fee) / equity

        opincome = self.get_data('FS_Comins', 'B001300000', lag_rule='lag_rule2')
        book_value = self.get_data('FS_Combas', 'A003000000', lag_rule='lag_rule1')
        operprof = opincome / book_value
        
        return operprof
    
    # % change in gross margin - %change in sales, pchgm_pchsale
    def calc_pchgm_pchsale(self):
        
        opincome = self.get_data('FS_Comins', 'B001300000', lag_rule='lag_rule2')
        opre = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        # opcost = self.get_data('FS_Comins', 'B001201000', lag_rule='lag_rule2')
        gross_margin = opincome/ opre
        ave_gross_margin = (gross_margin.shift(12) + gross_margin.shift(24)) / 2
        pchgm = (gross_margin - ave_gross_margin) / ave_gross_margin
        # ave_opincome = (opincome.shift(12) + opincome.shift(24)) / 2
        # pchgm = (opincome - ave_opincome) / ave_opincome
        # pchgm = (chgm - chgm.shift(12)) / chgm.shift(12)
        # sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        # pchsale = (sale - sale.shift(12)) / sale.shift(12)
        pchsale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2', pch=True)
        pchgm_pchsale = pchgm - pchsale
        
        return pchgm_pchsale
    
    # TIP Done perfectly
    # Asset turnover, ATO
    def calc_ato(self):

        noa = self.calc_noa()
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        ato = sale / noa
        
        return ato


    # Number of earnings increases, nincr
    
    # Return on invested capital, roic
    def calc_roic(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        liab = self.get_data('FS_Combas', 'A002000000', lag_rule='lag_rule2')
        equity = self.get_data('FS_Combas', 'A003000000', lag_rule='lag_rule2')
        cash = self.get_data('FS_Combas', 'A001101000', lag_rule='lag_rule2')
        roic = sale / (liab + equity - cash)
        
        return roic
    
    # Revenue surprise, rusp
    def calc_rsup(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd', lag=True, lag_rule='lag_rule2')
        market_equity = market_equity * 1000
        rsup = (sale - sale.shift(4)) / market_equity
        
        return rsup
    
    
    ##  财务流动因子
    # TIP Done perfectly
    # current ratio, CR
    def calc_cr(self):
        
        current_assets = self.get_data('FS_Combas', 'A001100000', lag_rule='lag_rule2')
        current_liab = self.get_data('FS_Combas', 'A002100000', lag_rule='lag_rule2')
        cr = current_assets / current_liab
        
        return cr
    
    # TIP Done perfectly
    # quick ratio, QR
    def calc_qr(self):
        
        current_assets = self.get_data('FS_Combas', 'A001100000', lag_rule='lag_rule2')
        inventory = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule2')
        current_liab = self.get_data('FS_Combas', 'A002100000', lag_rule='lag_rule2')
        qr = (current_assets - inventory) / current_liab
        
        return qr
    
    # TIP Done perfectly
    # cashdebt
    def calc_cashdebt(self):
    
        ttl_liab = self.get_data('FS_Combas', 'A002000000', lag_rule='lag_rule2')
        ave_liab = (ttl_liab + ttl_liab.shift(12)) / 2
        net_income = self.get_data('FS_Comins', 'B002000000', lag_rule='lag_rule2')
        cashdeta = net_income / ave_liab
    
        return cashdeta
    
    # TIP Done perfectly
    # sales to cash ratio, salecash
    def calc_salecash(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        cash = self.get_data('FS_Combas', 'A001101000', lag_rule='lag_rule2')
        universe = np.isnan(sale)
        salecash = sale / cash
        salecash.replace({np.inf:np.nan}, inplace=True)
        salecash.fillna(method='ffill', inplace=True)
        salecash[universe] = np.nan

        return salecash
    
    # TIP Done perfectly
    # sales to inventory ratio, saleinv
    def calc_saleinv(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        inv = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule2')
        universe = np.isnan(sale)
        # inv.replace({0:np.nan}, inplace=True)
        # inv.fillna(method='ffill', inplace=True)
        # inv[universe] = np.nan
        saleinv = sale / inv
        saleinv.replace({np.inf:np.nan}, inplace=True)
        saleinv.fillna(method='ffill', inplace=True)
        saleinv[universe] = np.nan

        return saleinv
    
    # TIP Done perfectly
    # current ratio growth, CRG
    def calc_crg(self):
        
        cr = self.calc_cr()
        crg = (cr - cr.shift(12)) / cr.shift(12)
        
        return crg
    
    # TIP Done perfectly
    # quick ratio growth, QRG
    def calc_qrg(self):
        
        qr = self.calc_qr()
        qrg = (qr - qr.shift(12)) / qr.shift(12)
        
        return qrg
    
    # TIP Done perfectly
    # % change sales-to-inventory, pchsaleinv
    def calc_pchsaleinv(self):
        
        saleinv = self.calc_saleinv()
        pchsaleinv = (saleinv - saleinv.shift(12)) / saleinv.shift(12)
        
        return pchsaleinv
    
    # TIP Done perfectly
    # Sales to receivables, salerec
    def calc_salerec(self):
        
        sale = self.get_data('FS_Comins', 'B001101000', lag_rule='lag_rule2')
        rec = self.get_data('FS_Combas', 'A001111000', lag_rule='lag_rule2')
        salerec = sale / rec 
        universe = np.isnan(sale)
        salerec = sale / rec
        salerec.replace({np.inf:np.nan}, inplace=True)
        salerec.fillna(method='ffill', inplace=True)
        salerec[universe] = np.nan

        return salerec
    
    # TIP Done perfectly
    # Debt capacity/firm tangibility, tang
    def calc_tang(self):
        
        cash = self.get_data('FS_Combas', 'A001101000', lag_rule='lag_rule2')
        rec = self.get_data('FS_Combas', 'A001111000', lag_rule='lag_rule2')
        inv = self.get_data('FS_Combas', 'A001123000', lag_rule='lag_rule2')
        fixed_assets = self.get_data('FS_Combas', 'A001212000', lag_rule='lag_rule2')
        ttl_assets = self.get_data('FS_Combas', 'A001000000', lag_rule='lag_rule2')
        tang = (cash + 0.715 * rec + 0.547 * inv + 0.535 * fixed_assets) / ttl_assets
        
        return tang
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    