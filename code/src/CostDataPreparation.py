# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:33:21 2019

@author: Mulugeta W.Asres

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc
import numpy as np,pandas as pd
import time
from scipy import signal

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# %%
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_path)
import utilities as util

base_path = os.path.dirname(os.getcwd())
code_path, data_path, main_result_path = util.GetWorkingDirs(base_path) 
sys.path.append(base_path)
sys.path.append(code_path)

setting_dict  = util.LoadJson("{}/setting.json".format(src_path))

# %%
FIGURE_WIDTH = 6.4
FIGURE_HEIGHT = 4
FIGURE_LBL_FONTSIZE_MAIN = 12

util.SetFigureStyle(FIGURE_WIDTH=FIGURE_WIDTH, FIGURE_HEIGHT=FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN=FIGURE_LBL_FONTSIZE_MAIN)
kws_online_2 = util.kws_online_2
kws_box_2 = util.kws_box_2

# %% 
class DatasetPreparation:
    """
    Calculates the computational costs, i.e processing energy and processing time from the extracted ROIs and 
    prepares the cost datasets that will be used later for cost analysis and model development. 
    The processing time is estimated as the start and end time of the ROIs multiplied with the meter's sampling interval. 
    For the processing energy, a trapezoidal method is employed to approximate the integral function of the power from the discrete sample power readings. 
    """
    def __init__(self, result_path):
        
        _, result_path = util.GetTargetDir(result_path, "target log folder")
        print("Result directory is: " + result_path)
        self.result_path = result_path

        self.MPOpts = 'MP'
        self.df_meter_log_roi = None
        self.df_meter_log_roi_cost =  None
        self.P_threshold = 10 # 10W 
        self.meter_TS = setting_dict["meter_log"]["sampling_rate_sec"] #meter sampling timee.g. whattsup is 1 seconds
        self.exprmnt_setting_dict = None
        self.df_cost_agg = None
        self.df_cost = None
        self.data_agg_alg = np.median
        self.meter_log_path = "{}/{}".format(self.result_path, setting_dict["meter_log"]["log_dir"])
        self.system_log_path = "{}/{}".format(self.result_path, setting_dict["system_log"]["log_dir"])
        self.filename_exprmmntsetting = "{}/{}".format(self.result_path, setting_dict["meter_log"]["setting_file"])
        self.filename_meter_log_roi = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["extracted_file"])
        self.filename_costdataset = "{}/{}".format(self.result_path, setting_dict["cost_dataset"]["raw_file"])
        self.filename_costdataset_agg = "{}/{}".format(self.result_path, setting_dict["cost_dataset"]["agg_file"])      

        # print("result_path: ", self.result_path)
        # print("filename_exprmmntsetting: ", self.filename_exprmmntsetting)
        # print("filename_meter_log_roi: ", self.filename_meter_log_roi)
    
    def LoadExprimentSetting(self):
        
        self.exprmnt_setting_dict = util.LoadJson(self.filename_exprmmntsetting)
        util.PrintDict(self.exprmnt_setting_dict)

        self.isdays = self.exprmnt_setting_dict['isdays']
        self.disaggregation_window = self.exprmnt_setting_dict['disaggtimewindow']
        self.num_repeated_expt = self.exprmnt_setting_dict['num_repeated_expt']
        self.num_Nh_chunks = self.exprmnt_setting_dict['num_Nh_chunks']
        self.num_Nps = self.exprmnt_setting_dict['num_Nps']
        self.Nhs = self.exprmnt_setting_dict['Nh_chunks']
        self.Nps = self.exprmnt_setting_dict['Nps']
        # self.MPOpts = setting_dict['isMultiProc']
        self.MPOpts = "MP"

    def LoadExtractedROIMeterData(self):
        
        print("Loading  Meter Log Data... ")

        print("filename_meter_log_roi: ", self.filename_meter_log_roi)
        self.df_meter_log_roi = pd.read_csv(self.filename_meter_log_roi)

        if self.df_meter_log_roi.empty:
            print('error: empty data:df_meter_log_roi!')
            return
        
        self.df_meter_log_roi.onlineId = self.df_meter_log_roi.onlineId + 1
        self.df_meter_log_roi.expId = self.df_meter_log_roi.expId + 1
    
    def PreprocessExtractedROIMeterData(self):
        
        def _CleanNoise(df_nilm):
        
            dfs = []
            expIds = list(df_nilm['expId'].unique())
            filter_size = 15
            for expId in expIds:
                sel_expId = df_nilm['expId'] == expId
                NhIds = list(df_nilm.loc[sel_expId, 'NhId'].unique())
                
                for NhId in NhIds:
                    sel_NhId = sel_expId & (df_nilm['NhId'] == NhId)
                    NpIds = list(df_nilm.loc[sel_NhId, 'NpId'].unique())
                    
                    for NpId in NpIds:
                        sel_NpId = sel_NhId & (df_nilm['NpId'] == NpId)
                        df = df_nilm.loc[sel_NpId, :]
                        P = signal.medfilt(df['P_w'], kernel_size=filter_size)
                        P_med = np.max(P)                                     
                        P_th = 0.35*P_med

                        idx_start = df.index[0]
                        df = df.loc[P >= P_th, :].reset_index(drop=True)
                        df.index = df.index + idx_start
                       
                        dfs.append(df)
                        
            df_nilm = pd.concat(dfs, axis = 'rows', ignore_index=True)
            return df_nilm
            
        def _EnergyEstimation(x):    
            
            def _Trapezoidal(f, a, b, n):
                h = (b-a)/float(n)
                I = f[a] + f[b]
                for k in range(1, n, 1):
                    t = int(a + k*h)
                    I += 2*f[t]
                I *= h/2
                return I
            
            f = x.values
            n = len(f) - 1
            a = 0
            b = n
            
            return _Trapezoidal(f, a, b, n)

        print("Preprocessing Meter Log Data... ")
        self.df_meter_log_roi = _CleanNoise(self.df_meter_log_roi)
        
        self.df_meter_log_roi = self.df_meter_log_roi[['t', 'P_w', 'P_bg_w', 'expId', 'NhId', 'NpId', 'onlineId']]
        self.df_meter_log_roi['t'] = [self.meter_TS]*self.df_meter_log_roi.shape[0]

        nilm_P_F = signal.medfilt(self.df_meter_log_roi['P_w'], kernel_size=3)
        self.df_meter_log_roi['P_w'] =  nilm_P_F 
        self.df_meter_log_roi = self.df_meter_log_roi.loc[self.df_meter_log_roi['P_w'] > self.P_threshold, :] 
        P_bg = self.df_meter_log_roi['P_bg_w'].iloc[0]
        print("Background Power (W): ", P_bg)

        df_grp = self.df_meter_log_roi.groupby(['NhId', 'expId', 'NpId', 'onlineId'], as_index=False)
        df_meter_log_roi_cost = df_grp.aggregate(np.sum)
        power_cols = ['P_w', 'P_bg_w']
        df_meter_log_roi_cost[power_cols] = df_grp.aggregate(_EnergyEstimation)[power_cols]
        df_meter_log_roi_cost.shape
        
        df_meter_log_roi_cost['energy_nilm_wh'] = df_meter_log_roi_cost['P_w']/(3600)
        df_meter_log_roi_cost['energy_bg_wh'] = df_meter_log_roi_cost['P_bg_w']/(3600)
        df_meter_log_roi_cost['time_peronline_total_sec'] = df_meter_log_roi_cost['t'].copy()

        df_meter_log_roi_cost['power_bg_median_w'] = (df_meter_log_roi_cost['energy_bg_wh']*(3600))/df_meter_log_roi_cost['t']
        df_meter_log_roi_cost['power_nilm_median_w'] = (df_meter_log_roi_cost['energy_nilm_wh']*(3600))/df_meter_log_roi_cost['t']

        df_meter_log_roi_cost['energy_wh'] = df_meter_log_roi_cost['energy_nilm_wh'] + df_meter_log_roi_cost['energy_bg_wh']

        df_meter_log_roi_cost['energy_bg_dw_wh'] = df_meter_log_roi_cost['power_bg_median_w']*self.disaggregation_window*60/(3600)

        df_naec_median = df_grp.aggregate(self.data_agg_alg)

        df_meter_log_roi_cost['power_median_w'] = df_meter_log_roi_cost['power_nilm_median_w'] + df_meter_log_roi_cost['power_bg_median_w']
        df_meter_log_roi_cost['power_utilization_median_perc'] = 100*df_meter_log_roi_cost['power_nilm_median_w']/df_meter_log_roi_cost['power_median_w']
        df_meter_log_roi_cost.drop(columns=['t', 'P_w', 'P_bg_w'], inplace=True)

        df_meter_log_roi_cost.columns = df_meter_log_roi_cost.add_prefix('meter_').columns

        mapdict = dict()
        for i, Nh in enumerate(self.Nhs):
            mapdict[i] = Nh
        df_meter_log_roi_cost['meter_NhId'] = df_meter_log_roi_cost['meter_NhId'].map(mapdict)
        df_meter_log_roi_cost.rename(columns={
                                'meter_expId':'expId',
                                'meter_NhId':'num_houses',       
                                'meter_NpId':'num_processes', 
                                'meter_onlineId':'onlineId'
                            }, inplace=True)

        self.df_meter_log_roi_cost = df_meter_log_roi_cost
        # print(self.df_meter_log_roi_cost.shape)
    
    def LoadSystemLogData(self):
        
        print("Loading System Log Data... ")
        
        DisagWindows = [self.disaggregation_window]
    
        dfs = []
        PO = self.MPOpts
        
        # for expId in range(0, self.num_repeated_expt):
        for expId in util.tqdm(range(self.num_repeated_expt), ncols=100):
            
            # print("expId: ", expId)

            for Nh in self.Nhs:
                # print("NumberHousePerMp: ", Nh)
                for Np in self.Nps:    
                    # print("Parallel Processing Np: ", Np)
                    
                    for dw in DisagWindows:
                        # print("DisagWindows: ", dw)
                        # df.head()
                        filename = util.resource_log_filename_format.format(self.system_log_path, PO, dw, expId, Nh, Np)
    
                        df = pd.read_csv(filename+"_per_sys.csv")
                        
                        df['expId'] = expId + 1
                        df['time_peronline_total_sec'] = df[['time_mpInit_sec','time_disagg_online_sec', 'time_mpClose_sec']].sum(axis=1)
                        df['time_mp_total_sec'] = df[['time_mpInit_sec', 'time_mpClose_sec']].sum(axis=1)

                        df_proc = pd.read_csv(filename+"_per_process.csv")
                        # df_proc['energy_wh'] = df_proc['energy_kwh'].apply(lambda x: 1000.0*x)
                        df['house_energy_min_kwh'] = df_proc['energy_kwh'].min(axis=0)
                        df['house_energy_mean_kwh'] = df_proc['energy_kwh'].mean(axis=0)
                        df['house_energy_median_kwh'] = df_proc['energy_kwh'].median(axis=0)
                        df['house_energy_max_kwh'] = df_proc['energy_kwh'].max(axis=0)
                        df['house_energy_total_kwh'] = df_proc['energy_kwh'].sum(axis=0)

                        dfs.append(df)
        
        df_system_log_pt = pd.concat(dfs, axis=0, ignore_index=True)
        df_system_log_pt.rename(columns = {'numProcesses': 'num_processes'}, inplace=True)
        # print(df_system_log_pt.shape)
        # print(df_system_log_pt.head())
        self.df_system_log_pt = df_system_log_pt

    def PrepareCostDataset(self):
        
        exprmntIds_disagg = self.df_meter_log_roi_cost["expId"].unique()
        # print('disaggregated meter exprmntIds: ', exprmntIds_disagg)
        exprmntIds_status = self.df_system_log_pt.expId.apply(lambda x: x in exprmntIds_disagg)
        df_system_log_pt = self.df_system_log_pt.loc[exprmntIds_status, :]
        # print('cleaned exprmntIds: ', df_system_log_pt.expId.unique())
        self.df_cost = pd.merge(df_system_log_pt, self.df_meter_log_roi_cost, how='right', on=['expId', 'num_houses', 'num_processes', 'onlineId'])
        # print(self.df_cost.head())
        # print(self.df_cost.shape)
        # print(self.df_cost.columns)

        sel_cols_online = [ 'num_processes', 'dw_min', 'num_houses', 'num_houses_chunk',
                            'num_houses_online', 'onlineId', 'expId',
                                        
                            'time_mpInit_sec',
                            'time_disagg_online_sec', 'time_mpClose_sec', 
                                        
                        #        'sys_cpu_t.user_d_sec',
                        #        'sys_cpu_t.sys_d_sec', 'sys_cpu_t.us_d_sec', 'sys_cpu_t.idle_d_sec',
                        #        'sys_cpu_t.interrupt_d_sec', 'sys_cpu_t.dpc_d_sec', 'sys_cpu_perc_d',
                        #        'sys_memvir.used_d_MB', 'sys_memvir.perc_d', 'sys_memswap.used_d_MB',
                        #        'sys_memswap.perc_d', 'sys_diskio.rd_d_cnt', 'sys_diskio.wr_d_cnt',
                        #        'sys_diskio.rd_d_MB', 'sys_diskio.wr_d_MB', 'sys_diskio.rd_t_d_sec',
                        #        'sys_diskio.wr_t_d_sec', 
                                
                            'time_total_online_sec', 'time_mpInit_total_sec',
                            'time_mpClose_total_sec', 'time_disagg_total_sec',
                            'time_peronline_total_sec', 'time_mp_total_sec',
                            'meter_time_peronline_total_sec', 
                            'meter_energy_nilm_wh', 'meter_energy_bg_wh','meter_energy_wh', 'meter_energy_bg_dw_wh',
                            'meter_power_bg_median_w', 'meter_power_nilm_median_w', 'meter_power_median_w', 'meter_power_utilization_median_perc',
                            # 'time_preProcessing_sec', 'time_loadSetting_sec',
                            # 'time_loadModel_sec', 'time_disagg_sec', 
                            # 'p_time_disagg_total_sec',
                            # 'energy_house_wh'

                            'house_energy_min_kwh','house_energy_mean_kwh', "house_energy_median_kwh", 'house_energy_max_kwh','house_energy_total_kwh'
                            ]
        
        self.df_cost[sel_cols_online[6:]] = self.df_cost[sel_cols_online[6:]].round(4)
        self.df_cost_agg = self.df_cost[sel_cols_online].groupby(sel_cols_online[:6], as_index=False).aggregate(self.data_agg_alg)
        self.df_cost_agg.drop(columns=['expId'], inplace=True)
        # print(self.df_cost_agg.shape)

        cat_cols = ['expId','num_processes', 'dw_min', 'num_houses', 'num_houses_chunk', 'num_houses_online', 'onlineId']
        for col in cat_cols:
            self.df_cost[col] = self.df_cost[col].astype('category')
    
    def SaveCostDataset(self):
        
        util.SaveDatatoCSV(self.filename_costdataset, self.df_cost)
        util.SaveDatatoCSV(self.filename_costdataset_agg, self.df_cost_agg)

class DataAnalysis:
    """ provides some data analysis plots on the prepared cost dataset."""
    def __init__(self, result_path):

        self.result_path = result_path
        self.exprmnt_setting_dict = None
        self.meter_log_path = "{}/{}".format(self.result_path, setting_dict["meter_log"]["log_dir"])
        self.filename_exprmmntsetting = "{}/{}".format(self.result_path, setting_dict["meter_log"]["setting_file"])
        self.filename_costdataset = "{}/{}".format(self.result_path, setting_dict["cost_dataset"]["raw_file"])
        self.filename_costdataset_agg = "{}/{}".format(self.result_path, setting_dict["cost_dataset"]["agg_file"]) 
        self.figure_path = "{}/{}".format(self.result_path, setting_dict["meter_log"]["figure_dir"])
            
        self.data_agg_alg = np.mean
        self.plot_sel_cols_online = [ 
                                    # 'time_mp_total_sec',
                                    # 'time_mpInit_sec',
                                    # 'time_disagg_online_sec', 
                                    # 'time_mpClose_sec', 
                                    # 'time_peronline_total_sec', 
                                    # 'sys_cpu_t.user_d_sec',
                                    # 'sys_cpu_t.sys_d_sec', 'sys_cpu_t.us_d_sec', 'sys_cpu_t.idle_d_sec',
                                    # 'sys_cpu_t.interrupt_d_sec', 'sys_cpu_t.dpc_d_sec', 'sys_cpu_perc_d',
                                    # 'sys_memvir.used_d_MB', 'sys_memvir.perc_d', 'sys_memswap.used_d_MB',
                                    # 'sys_memswap.perc_d', 
                                    # 'sys_diskio.rd_d_cnt', 'sys_diskio.wr_d_cnt',
                                    # 'sys_diskio.rd_d_MB', 'sys_diskio.wr_d_MB', 'sys_diskio.rd_t_d_sec','sys_diskio.wr_t_d_sec',
                                    
                                    # 'time_total_online_sec', 
                                    # 'time_mpInit_total_sec',
                                    # 'time_mpClose_total_sec', 
                                    'time_disagg_total_sec',
                                    'time_peronline_total_sec', 
                                    # 'time_mp_total_sec',
                                    
                                    'meter_time_peronline_total_sec', 
                                    'meter_energy_nilm_wh', 
                                    # 'meter_energy_bg_wh',
                                    # 'meter_energy_wh', 
                                    # 'meter_energy_bg_dw_wh', 
                                    # 'meter_power_bg_median_w', 
                                    # 'meter_power_nilm_median_w',
                                    # 'meter_power_median_w', 
                                    # 'meter_power_utilization_median_perc', 
                                    
                                    # 'time_preProcessing_sec', 'time_loadSetting_sec',
                                    # 'time_loadModel_sec', 'time_disagg_sec', 
                                    # 'p_time_disagg_total_sec',
                                    # 'energy_house_wh'
                                    ]
        self.sel_cols_online = ['num_processes', 'dw_min', 'num_houses', 'num_houses_chunk',
                            'num_houses_online', 'onlineId', 'expId'
                                        
                            # 'time_mpInit_sec',
                            # 'time_mpClose_sec', 
                            # 'time_disagg_online_sec',

                            # 'time_total_online_sec', 
                            # 'time_mpInit_total_sec',
                            # 'time_mpClose_total_sec', 
                            'time_disagg_total_sec',
                            'time_peronline_total_sec', 
                            # 'time_mp_total_sec',

                            'meter_time_peronline_total_sec', 

                            'meter_energy_nilm_wh', 
                            'meter_energy_bg_wh',
                            'meter_energy_wh',
                            
                            'meter_energy_bg_dw_wh',
                            'meter_power_bg_median_w', 
                            'meter_power_nilm_median_w', 
                            'meter_power_median_w', 
                            'meter_power_utilization_median_perc',
                                ]
    
    def LoadCostDataset(self):
        
        print("loading costdatasets...")
        df_cost = pd.read_csv(self.filename_costdataset)
        df_cost_agg = pd.read_csv(self.filename_costdataset_agg)
        
        if df_cost.empty:
            print('error: empty data:df_cost!')
        
        if df_cost_agg.empty:
            print('error: empty data:df_cost_agg!')
        try:
            for col in self.plot_sel_cols_online:
                df_cost[col] = df_cost[col].astype('float')
                df_cost_agg[col] = df_cost_agg[col].astype('float')
            
            df_cost = df_cost.loc[
                ~(df_cost["num_houses"].isin(setting_dict["modeling"]["exclude_data"]["num_houses"]) | 
                df_cost["num_processes"].isin(setting_dict["modeling"]["exclude_data"]["num_processes"])), :]
                
            df_cost_agg =  df_cost_agg.loc[
                ~(df_cost_agg["num_houses"].isin(setting_dict["modeling"]["exclude_data"]["num_houses"]) | 
                df_cost_agg["num_processes"].isin(setting_dict["modeling"]["exclude_data"]["num_processes"])), :]
            
            print("costdataset size: {}".format(df_cost.shape))
        

        except Exception as ex:
            print("LoadCostDataset error: {}".format(ex))
        return df_cost, df_cost_agg
    
    def PlotAnalysis(self, df_cost, df_cost_agg, result_path, isfigshow=False):
        
        print('plot analysis....')
        
        util.SetFigureStyle()

        # %%
        y_vars = [
                'time_peronline_total_sec', 
                'meter_time_peronline_total_sec',

                'meter_energy_nilm_wh', 
                'meter_energy_bg_wh', 
                'meter_energy_wh', 
                # 'meter_energy_bg_dw_wh', 

                'meter_power_nilm_median_w',
                'meter_power_bg_median_w',
                'meter_power_median_w', 
                # 'meter_power_utilization_median_perc', 
                # 'energy_house_wh'
                ]
               
        x_var = 'num_processes'
        hue ='num_houses'    
        for y_var in y_vars:
            fig, ax = plt.subplots()
            fig = sns.catplot(x=x_var, y=y_var, data=df_cost, hue=hue, kind="point",  **kws_online_2)
            filename_plot = "{}/{}_vs_{}".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)

        df_cost["time_peronline_total_meter_vs_sys_diff_sec"] = df_cost["meter_time_peronline_total_sec"] - df_cost["time_peronline_total_sec"]
        df_cost["time_peronline_total_meter_vs_sys_diff_perc"] = 100*df_cost["time_peronline_total_meter_vs_sys_diff_sec"].divide(df_cost["time_peronline_total_sec"], axis="index").copy()
        for y_var in ["time_peronline_total_meter_vs_sys_diff_sec", "time_peronline_total_meter_vs_sys_diff_perc"]:
            fig, ax = plt.subplots()
            fig = sns.catplot(x=x_var, y=y_var, data=df_cost, hue=hue, kind="point",  **kws_online_2)
            filename_plot = "{}/{}_vs_{}".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)

        
        # %%
        # Normalization: perunit house
        
        y_vars = [y_var for y_var in y_vars if (y_var.find('power') == -1) and (y_var.find('diff') == -1)]
        print('Plotting normalized per unit house values ...')

        df_cost_norm = df_cost.copy()
        norm_var = 'num_houses'
        df_cost_norm[norm_var] = df_cost_norm[norm_var].astype('float')
        for col in self.sel_cols_online[7:]:
            df_cost_norm[col] = df_cost_norm[[col]].divide(df_cost_norm["num_houses"], axis="index")
        df_cost_norm[norm_var] = df_cost_norm[norm_var].astype('category')

        x_var ='num_processes'
        hue = 'num_houses'
        for y_var in y_vars:
            fig, ax = plt.subplots()
            fig = sns.catplot(x=x_var, y=y_var, data=df_cost_norm, hue=hue, kind="point",  **kws_online_2)
            filename_plot = "{}/{}_vs_{}_norm".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)
            
        # Variations on Experiments
        x_var = 'num_processes'
        hue = 'expId'
        col = 'num_houses'
        n_col = df_cost_norm[col].nunique()
        for y_var in y_vars:
            fig, ax = plt.subplots()
            fig = sns.catplot(x=x_var, y=y_var, data=df_cost_norm, col=col, hue=hue, kind="point", col_wrap=np.min([int(np.sqrt(n_col)), 4]), **kws_online_2)
            filename_plot = "{}/{}_vs_{}_all_exps_norm".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)

@util.timer
def main():
    
    util.clear()

    # select and set target system and cost log
    system_name, result_path = util.GetTargetDir(main_result_path, "target system")
    print("Result directory is: " + result_path)

    print('cost dataset prepration...')
    # Initialization of cost dataset preparation module 
    ObjDatasetPreparation = DatasetPreparation(result_path)
    
    # load cost monitoring settings
    ObjDatasetPreparation.LoadExprimentSetting()
    
    # load extracted ROIs power measurements for the NILM algorithm 
    ObjDatasetPreparation.LoadExtractedROIMeterData()
    
    # estimation of processing time and energy costs
    ObjDatasetPreparation.PreprocessExtractedROIMeterData()
    
    # load processing time cost estimation from in code measurement from system logs
    ObjDatasetPreparation.LoadSystemLogData()
    
    # prepare cost datasets from meter and system logs
    ObjDatasetPreparation.PrepareCostDataset()
    
    # store cost dataset for later use
    ObjDatasetPreparation.SaveCostDataset()

    print('cost dataset illustrations....')
    # Initialization of data analysis 
    ObjDataAnalysis = DataAnalysis(ObjDatasetPreparation.result_path)
    
    # load cost dataset
    df_cost, df_cost_agg = ObjDataAnalysis.LoadCostDataset()
    
    # illustrate the cost dataset in data analysis plots
    ObjDataAnalysis.PlotAnalysis(df_cost, df_cost_agg, result_path)
    
    msg = "{}\n The resulting cost datasets can be found in {}/ and \n some illustrations are placed in {}/.\n{}"\
            .format("*"*60, setting_dict["meter_log"]["log_dir"], setting_dict["meter_log"]["figure_dir"], "*"*60)
    print(msg)

    print('Done!')
    
if __name__ == '__main__': 

    main()