"""
Created on Fri Aug 30 17:33:21 2019

@author: Mulugeta W.Asres

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc
import numpy as np, pandas as pd
import datetime, time
from itertools import groupby
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize
        
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
 
# %%
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_path)
import utilities as util

base_path = os.path.dirname(os.getcwd())
code_path, data_path, result_path = util.GetWorkingDirs(base_path) 
sys.path.append(base_path)
sys.path.append(code_path)

setting_dict  = util.LoadJson("{}/setting.json".format(src_path))

# import MeterReadingROIPlotter as ROIPlotter

if not os.path.exists(result_path):
    os.makedirs(result_path)

#%% plot settings
isdebug = 0 # 1 for detailed debegging info in case of need to see cause corrupted ROIs
# %%
FIGURE_WIDTH = 6.4
FIGURE_HEIGHT = 2
FIGURE_LBL_FONTSIZE_MAIN = 12

util.SetFigureStyle(FIGURE_WIDTH, FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN)

# %%
class AlgCostSignalExtractor:
    """
    Provides tools for hierarchical and recursive extraction the region of interest (ROI) sections from the power reading log loaded from a power meter. 
    This module uses the cluster-based synchronization signal extraction technique to capture the ROIs. 
    There are four major ROIs: 
    >>> NILM-ROI corresponds to the entire experiment of target algorithm executions after isolation of the static and background power noises from the power log. 
    >>> Repetition-ROIs are sections of each experiment repetitions after segregation of NILM-ROI. 
    >>> LoadScale-ROIs contain regions belong to executions on given load scales from each Repetition-ROI.
    >>> MultiProcess-ROIs correspond to regions of execution of particular number of parallel processors selections for a given LoadScale-ROI.
    """
    def __init__(self, result_path):
        
        self.df_wupower_log = None
        self.df_wupower_log_disaggregated = None
        self.exprmnt_setting_dict = None
        self.corrupted_expIds = []
        # self.filename_errorlog  = base_path+"/results/" + system_name+"/WU_pattern_extraction_error_log.txt"
       
        self.result_path = result_path  
        self.meter_log_path = "{}/{}".format(self.result_path, setting_dict["meter_log"]["log_dir"])
        self.filename_exprmmntsetting = "{}/{}".format(self.result_path, setting_dict["meter_log"]["setting_file"])
        self.filename_meter_log = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["log_file"])
        # self.filename_meter_log = "{}/wattsup_log/{}".format(self.result_path, filename_meter_log)
        self.filename_meter_log_roi_uncleaned = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["uncleaned_extracted_file"])
        self.filename_meter_log_roi = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["extracted_file"])
        self.filename_errorlog = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["errorlog_file"])
        
        self.figure_path = "{}/{}".format(self.meter_log_path, setting_dict["meter_log"]["debug_figure_dir"])
        util.CreateDir(self.figure_path)
        
        # print("result_path: ", self.result_path)
        # print("filename_exprmmntsetting: ", self.filename_exprmmntsetting)
        # print("filename_meter_log: ", self.filename_meter_log)
        # print("filename_errorlog: ", self.filename_errorlog)
        # print("filename_meter_log_roi: ", self.filename_meter_log_roi)
    
    def _CheckNumExtractedSections(self, var_name, num_expected, num_extracted, extracted_roi_durations, sourceId=None):
        
        try:
            assert(num_expected == num_extracted), "The number of expected sections ({}) is not equal to number of extracted ({}). Run debug mode by setting isdebug = 1!".format(num_expected, num_extracted)  
        except AssertionError as error:
            # print('number of the extracted sections does not match with the expected: ', [num_expected, num_extracted])
            outliers = self._OutlierDetection(extracted_roi_durations)
            msg_dict = dict()
            msg_dict[var_name+'_counts'] = extracted_roi_durations.to_dict()
            msg_dict['error_message'] = str(error)
            msg_dict['outlier_' + var_name] = outliers
            msg_dict['corrupted_sourceId'] = outliers
            msg_dict['corrupted_expId'] = outliers

            if isinstance(sourceId, dict):
                msg_dict['corrupted_sourceId'] = sourceId
                msg_dict['corrupted_expId'] = sourceId['expId']
                # print(msg_dict)

            self._AddToErrorLog(msg_dict)       
    
    def _AddToErrorLog(self, msg_dict):
        
        print("\n error is raised and logged into WU_pattern_extraction_error_log.txt")
        
        if isinstance(msg_dict['corrupted_expId'], list):
            self.corrupted_expIds.extend(msg_dict['corrupted_expId']) 
        else:
            self.corrupted_expIds.append(msg_dict['corrupted_expId']) 
        
        log = str(msg_dict)
        
        with open(self.filename_errorlog, 'a') as fhandle:
            fhandle.write("{}{}\n".format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S "), log))
            fhandle.close()
    
    def _OutlierDetection(self, df):
        
        mu = df.mean()
        std = df.std()
        isoutlier = df.apply(lambda x: np.abs(x - mu) > std)
        outliers = list(df.index[isoutlier].values)
        
        return outliers
   
    def _CleanNILMROI(self, df_nilm, post_alg="threshold"):

        print('Cleaning and optmizing start and end locations ...')
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
                    P = signal.medfilt(df['Watts'], kernel_size=filter_size)
                    d = len(P)
                    P_med = np.max(P)                                     
                    P_std = df['Watts'].iloc[int(0.25*d):int(0.75*d)].std() 
                        
                    if post_alg == "threshold":
                                             
                        P_th = 0.20*P_med
                            
                        v_idx = list(df.loc[P < P_th, :].index)
                        idx_med = (df.index[0] + df.index[-1])/2
                        
                        if (len(v_idx) >= 2) and (v_idx[0] < idx_med) and (v_idx[-1] > idx_med):           
                            v_idx_d = np.diff(v_idx)
                            v_idx_d_max_idx = np.argmax(v_idx_d)
                            df = df.loc[v_idx[v_idx_d_max_idx]:v_idx[v_idx_d_max_idx+1], :]
                            # print("adjusted:", (expId, NhId, NpId, P_med, P_std, P_th, len(P), df.shape))                    
                        
                        
                        # idx_start = df.index[0]
                        # df = df.loc[P >= 0.20*P_med, :].reset_index(drop=True)
                        # df.index = df.index + idx_start
                    
                    elif post_alg == "outlier":
                        
                        X = P.reshape(-1, 1)
                        # X = df['Watts'].values.reshape(-1, 1)                   
                        # X = np.concatenate((X, np.arange(0, d).reshape(-1, 1)), axis=1) 
                        
                        X = normalize(X, axis=0)

                        n_neighbors = int(0.01*d)
                        contamination = 10/d + 0.01
                        clf = IsolationForest(n_estimators=10, contamination=contamination, random_state=10).fit(X)
                        # clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(X)                       
                    
                        fig, ax = plt.subplots()
                        df['Watts'].plot()
                        
                        idx_med = 0.5*d
                        idx = np.arange(0, d)
                              
                        p_idx = idx[clf.predict(X) == 1] # idx[(clf.predict(X) == 1) & (P > 0.35*P_med)]
                        # v_idx = idx[clf.predict(X) == -1]
                        if len(p_idx) >= 2:
                            roi_idx_start = p_idx[0]
                            roi_idx_end = p_idx[-1]
                            df = df.iloc[roi_idx_start:roi_idx_end, :]
                        
                        # df['Watts'].plot()
                        # plt.title("{}".format(([expId, NhId, NpId, len(P), df.shape, contamination])))
                        # plt.show()                     
                    
                    dfs.append(df)
                    
              
        df_nilm = pd.concat(dfs, axis = 'rows', ignore_index=True)
        
        return df_nilm
   
    def ResetErrorLog(self):
        
        util.ResetLogFile(self.filename_errorlog)
        
    def LoadExprimentSetting(self):
        
        self.exprmnt_setting_dict = util.LoadJson(self.filename_exprmmntsetting)
      
    def LoadMeterPowerReading(self):
        
        print("filename_meter_log: ", self.filename_meter_log)
        self.df_wupower_log = pd.read_csv(self.filename_meter_log, delimiter='\t', names=['Time', 'Watts', 'Volts', 'Amps', 'WattHrs', 'Cost', 'Avg Kwh',
        'Mo Cost', 'Max Wts', 'Max Vlt', 'Max Amp', 'Min Wts', 'Min Vlt',
        'Min Amp', 'Pwr Fct', 'Dty Cyc', 'Pwr Cyc', 'Freq', 'Volts-Amps'], header=1)
        self.df_wupower_log.shape
        # self.df_wupower_log.head()
        if self.df_wupower_log.empty:
            print('error: empty data:df_wupower_log!')
            self._AddToErrorLog('LoadMeterPowerReading error: empty data:df_wupower_log.')
            return df_power_agg, None

        self.df_wupower_log = self.df_wupower_log[['Time', 'Watts', 'Volts', 'Amps']]
        self.df_wupower_log['t'] = np.arange(0, self.df_wupower_log.shape[0])
        self.df_wupower_log.head()


        filter_window = 5
        trend_window = 2*3*self.exprmnt_setting_dict['onlinedw_d'] + 1 #make it odd
        self.df_wupower_log['Watts_raw'] = self.df_wupower_log.Watts.copy()
        self.df_wupower_log['Watts_trend'] = self.df_wupower_log.Watts.rolling(trend_window).median() 
        self.df_wupower_log['Watts'] = self.df_wupower_log.Watts.rolling(filter_window).median() 
        self.df_wupower_log.head()
        self.df_wupower_log['Watts_trend'] = self.df_wupower_log['Watts_trend'].shift(periods=int(-0.5*trend_window))
        self.df_wupower_log['Watts'] = self.df_wupower_log['Watts'].shift(periods=int(-0.5*filter_window))
    
        self.bg_power = self.df_wupower_log.Watts.min()
        print('Estimated background power (watts): ', self.bg_power)
        
        fig, ax = plt.subplots()
        # plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts_raw, label= 'Total Power (Raw)')
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts, label= 'Total Power')
        # plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts_trend, label= 'Total Power (Trend)')

        self.df_wupower_log['Watts'] = self.df_wupower_log['Watts'] - self.bg_power
        self.df_wupower_log['Watts_trend'] = self.df_wupower_log['Watts_trend'] - self.bg_power
        self.df_wupower_log['Watts_bg'] = self.bg_power
        self.df_wupower_log.dropna(axis=0, inplace=True)
        self.df_wupower_log['t'] = np.arange(0, self.df_wupower_log.shape[0])
        
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts, label= 'Dynamic Power')
        plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        ax.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)   
        plt.legend()
        util.PlotGridSpacing(self.df_wupower_log.t.values, [0, self.df_wupower_log.Watts.max() + self.bg_power], x_gridno=6, y_gridno=6, issnscat = True)
        
        if isdebug == 1:
            plt.show()

        self.df_wupower_log = self.ClusterDataPoints(self.df_wupower_log)
        
        return self.df_wupower_log, fig
    
    def SaveExtractedData(self, df_disagg, iscleaned=1):
        
        if iscleaned == 1:
            filename = self.filename_meter_log_roi
            self.df_wupower_log_disaggregated = df_disagg.copy()
        else:
            filename = self.filename_meter_log_roi_uncleaned
        
        # print("filename_meter_log_roi: ", filename)
        df_disagg.rename(columns={'Watts':'P_w', 'Watts_bg':'P_bg_w'}, inplace=True)
        util.SaveDatatoCSV(filename, df_disagg)
    
    def ClusterDataPoints(self, df_wupower_log):
        
        x = df_wupower_log['Watts_trend'].values
        y, x = np.histogram(x, bins=30, density=True)
        norm_pmf = signal.medfilt(y, kernel_size=3)
        x = x[1:]

        width = int((np.max(x) - np.min(x))/10)
        
        while True:
            try:
                valley_indexes, _ = signal.find_peaks(-1*norm_pmf) #distance= width, width = 0.5*width
                self.synch_P_th = x[valley_indexes[0]]
                break
            except:
                width = int(width/2)
        target_x = x[valley_indexes[0]:]
        target_norm_pmf = norm_pmf[valley_indexes[0]:]
        isvalley_indexes_adj = target_norm_pmf <= 1.10*target_norm_pmf[0]
        self.synch_P_th = target_x[np.where(isvalley_indexes_adj==False)[0][0]-1]

        isvalley_indexes_adj[np.where(isvalley_indexes_adj==False)[0][0]:] = False
        valley_indexes = isvalley_indexes_adj

        fig, ax = plt.subplots()
        plt.plot(x, norm_pmf)
        plt.scatter(target_x[valley_indexes], target_norm_pmf[valley_indexes], marker='o', color='r', s=100)
        plt.plot([self.synch_P_th]*2, [0, np.max(norm_pmf)], linestyle='--', linewidth=2)
        plt.annotate("Decision Threshold {}".format(np.round(self.synch_P_th)), xy=(self.synch_P_th, 0.5*np.max(norm_pmf)), xytext=(self.synch_P_th+10, 0.75*np.max(norm_pmf)),
            bbox=dict(boxstyle="round", alpha=0.1),
            arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))
        plt.xlabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.ylabel('PDF', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)   
        util.PlotGridSpacing(x, [0, np.max(norm_pmf)], x_gridno=6, y_gridno=6, issnscat = True)
        plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)

        if isdebug:
            plt.show()
        util.SaveFigure("{}/meter_log_trend_PDF".format(self.figure_path), fig)

        self.df_wupower_log['isWUsynch'] = self.df_wupower_log['Watts_trend'] < self.synch_P_th
        self.df_wupower_log['isWUsynch'] = self.df_wupower_log['isWUsynch'].astype('int8')

        fig, ax = plt.subplots(figsize = [FIGURE_WIDTH, FIGURE_HEIGHT*3.1], nrows=3, ncols=1, constrained_layout=True)

        plt.subplot(311)
        
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts, linestyle='-', color='g', label = 'Dynamic Power')
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts_trend, linestyle='-', color='red', label = 'Dynamic Power (Trend)')
        plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)    
        util.PlotGridSpacing(self.df_wupower_log.t.values, [0, self.df_wupower_log.Watts.max()], x_gridno=6, y_gridno=6, issnscat = True)
        plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
        plt.legend(ncol = 2, loc='center right', bbox_to_anchor = (1, 1 + 0.1),   borderaxespad=0, frameon=True)
        
        
        plt.subplot(312)
        
        Ids = list(self.df_wupower_log["isWUsynch"].unique())
        for Id in Ids:
            df = self.df_wupower_log[self.df_wupower_log["isWUsynch"]==Id]
            if Id == 1:
                label = 'Cluster_Lower'
                marker = 'o'
            else:
                label = 'Cluster_Upper'
                marker = '*'
            plt.scatter(df.t, df.Watts_trend, label= label,  marker=marker)
        
        plt.axhline(y=self.synch_P_th, linestyle='-', color='black', label = "Threshold = watts".format(np.round(self.synch_P_th, 0)))
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts_trend, linestyle='--', color='red', label = 'Dynamic Power (Trend)')
        plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)   
        util.PlotGridSpacing(self.df_wupower_log.t.values, [0, self.df_wupower_log.Watts_trend.max()], x_gridno=6, y_gridno=6, issnscat = True)
        plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
        plt.legend(ncol = 2, loc='center right', bbox_to_anchor = (1, 1 + 0.1*2), borderaxespad=0, frameon=True)
        
        plt.subplot(313)
        
        for Id in Ids:
            df = self.df_wupower_log[self.df_wupower_log["isWUsynch"]==Id]
            if Id == 1:
                label = 'SynchPatterns'
                marker = 'o'
            else:
                label = 'Algorithm'
                marker = '*'
            plt.scatter(df.t, df.Watts, label= label,  marker=marker)
        
        plt.plot(self.df_wupower_log.t, self.df_wupower_log.Watts, linestyle='-', color='g', label = 'Dynamic Power')
        plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
        plt.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)   
        util.PlotGridSpacing(self.df_wupower_log.t.values, [0, self.df_wupower_log.Watts.max()], x_gridno=6, y_gridno=6, issnscat = True)
        plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
        plt.legend(ncol = 3, loc='center right', bbox_to_anchor = (1, 1 + 0.1), borderaxespad=0, frameon=True)
        
        if isdebug:
            plt.show()
        util.SaveFigure("{}/meter_log_clustered".format(self.figure_path), fig)

        return self.df_wupower_log

    def GetClusteredGroups(self, df):
        
        idx = df.index[0]
        pattern_grps_endidx = []
        pattern_grps = []
        ds_clusterIds = df['isWUsynch'].values
        for k, g in groupby(ds_clusterIds):
            g_size = len(list(g))
            try:
                if (g_size <= 5) | (k == pattern_grps[-1][0]): # to filter small noisy clusters groups and continate with previous grp
                    if g_size <= 5:
                        k = 1 - k

                    pattern_grps[-1][1] = pattern_grps[-1][1] + g_size 
                    idx = idx + g_size
                    pattern_grps_endidx[-1] = idx
                    continue
            except:
                pass
            pattern_grps.append([k, g_size, idx])
            idx = idx + g_size
            pattern_grps_endidx.append(idx)
            
        # print([df.index[0], df.index[-1], pattern_grps[0][2], pattern_grps_endidx[-1]])  
          
        return pattern_grps_endidx, pattern_grps
    
    def GetSynchLocations(self, pattern_grps, synch_d, expected_number_patterns=2, isinit=0, isdeep=0):
        
        synch_clusterId = 1
        found_pattern_startend_idx = []
        
        if isdeep == 0:
            for i in range(len(pattern_grps)):
                grpId = pattern_grps[i][0]
                if grpId == synch_clusterId:
                    duration = pattern_grps[i][1]
                    if duration > 0.95*synch_d:
                        found_pattern_startend_idx.append([pattern_grps[i][2], pattern_grps[i][2] + duration])
        else:
            pattern_grps = sorted(pattern_grps, key=lambda x: x[1], reverse=True)
            for i in range(len(pattern_grps)):
                grpId = pattern_grps[i][0]
                if grpId == synch_clusterId:
                    duration = pattern_grps[i][1]
                    if duration > 0.75*synch_d:
                        found_pattern_startend_idx.append([pattern_grps[i][2], pattern_grps[i][2] + duration])
            
        N_matched_patterns = len(found_pattern_startend_idx)
        if isinit == 0:
            found_pattern_startend_idx = found_pattern_startend_idx[0:np.min([N_matched_patterns, expected_number_patterns-1])]
            found_pattern_startend_idx = sorted(found_pattern_startend_idx, key=lambda x: x[0], reverse=False)
        N_matched_patterns = len(found_pattern_startend_idx)
        
        # print('found_pattern_startend_idx: ', found_pattern_startend_idx)
        # print('found_patterns: ', N_matched_patterns)
        
        return found_pattern_startend_idx
    
    def GetROISection(self, df, pattern_grps_endidx, end_idx, dP_binary=None, isplot=0, reindex=False, label='ROI'):
        
        df_section = df.loc[pattern_grps_endidx:end_idx,:] 
        fig = None
        
        if isplot:
            fig, ax = plt.subplots()
            plt.plot(df_section.t, np.array(df_section.Watts), label=label)
            plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
            plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
            ax.tick_params(axis='both', which='major', labelsize=FIGURE_LBL_FONTSIZE_MAIN)   
            plt.legend()
        
            if isdebug:
                plt.show()
            util.PlotGridSpacing(df_section.t.values, df_section.Watts.values, x_gridno=6, y_gridno=6, issnscat = True)
        
        if reindex:
            df_section.reset_index(drop=True, inplace=True)
            
        return df_section, fig 

    def ExtractROISections(self, df_power_agg, var_name, expected_number_patterns, d_synch_all, d_synch):
        
        def _ExtractROISections_main(df_power_agg, var_name, expected_number_patterns, d_synch_all, d):
            
            # print('expected_number_patterns: ', expected_number_patterns)
            if df_power_agg.empty:
                print('empty data:df_power_agg, may be due to incorrrect synch filtering in previous disaggregations!')
                return df_power_agg
            
            pattern_grps_endidx, pattern_grps = self.GetClusteredGroups(df_power_agg)
           
            if expected_number_patterns > 1: # to keep online synch patterns or just remove based on duration of online synch.
                found_pattern_startend_idx = self.GetSynchLocations(pattern_grps, d_synch_all, expected_number_patterns)
                N_matched_patterns = len(found_pattern_startend_idx)
    
                if  N_matched_patterns < expected_number_patterns-1:
                    N_missed_patterns = expected_number_patterns - N_matched_patterns-1
                    # print('partial pattern detection is running...N_missed_patterns:', N_missed_patterns)
                    found_pattern_startend_idx = self.GetSynchLocations(pattern_grps, d_synch_all, expected_number_patterns, isdeep=1)
                    N_matched_patterns = len(found_pattern_startend_idx)

                if var_name != 'onlineId':
                    var_idx = [[0, pattern_grps[0][2] + (5*d - 1)]]
                    d_synch_all = int(d_synch_all/2)                   
                    var_markers = [[found_pattern_startend_idx[i][0]+(d_synch_all-(5*d-1)), found_pattern_startend_idx[i][1]-d_synch_all+(5*d-1)] for i in range(len(found_pattern_startend_idx))]
                    var_idx.extend(var_markers)
                    var_idx.append([pattern_grps_endidx[-1]-(5*d - 1), pattern_grps_endidx[-1]])
                else:
                    var_idx = [[0, pattern_grps[0][2] + (5*d - 1)]]
                    var_markers = [[found_pattern_startend_idx[i][0], found_pattern_startend_idx[i][1]] for i in range(len(found_pattern_startend_idx))]
                    var_idx.extend(var_markers)
                    var_idx.append([pattern_grps_endidx[-1]-(5*d - 1), pattern_grps_endidx[-1]])
            else:
                var_idx = [[0, pattern_grps[0][2] + (5*d - 1)]]
                var_idx.append([pattern_grps_endidx[-1]-(5*d - 1), pattern_grps_endidx[-1]])
                
            # print(var_idx)   
            # print([df_power_agg.index[0], df_power_agg.index[-1]])
            dfs = []
            var_id_start = 0
            
            if var_name =='NpId':
                var_id_start = 1
            
            var_id = var_id_start
            
            for i in range(len(var_idx)-1):
                # print('var_id: ', var_id)
                df , _ = self.GetROISection(df_power_agg.copy(), var_idx[i][1], var_idx[i+1][0], isplot=0, reindex = False)
                df[var_name] = var_id
                var_id = var_id + 1     
                dfs.append(df)
            
            df_power_disagg = pd.concat(dfs, axis=0, ignore_index=False)
              
            if isdebug:
                # print(f'{var_name}={var_id - var_id_start}')
                plt.rcParams['figure.figsize'] = (15, 3)
                fig, ax = plt.subplots()
                plt.plot(df_power_agg.t, df_power_agg.Watts, label= 'P')
                p_max = df_power_disagg.Watts.max()
                for i in range(var_id_start, var_id):
                    df = df_power_disagg[df_power_disagg[var_name]==i]
                    plt.plot(df.t, df.Watts,  label= 'P_nilm_'+var_name+'_'+ str(i))
                plt.legend()
                plt.show()
                # time.sleep(1)
            
            return df_power_disagg
         
        df_power_disagg = _ExtractROISections_main(df_power_agg, var_name, expected_number_patterns, d_synch_all, d_synch)
       
        return df_power_disagg
    
    def NILMAlgROIExtraction(self, df_power_log, d_synch_all, d_synch):
        
        print('Extract NILM processing section....')
        
        pattern_grps_endidx, pattern_grps = self.GetClusteredGroups(df_power_log)
        
        found_pattern_startend_idx = self.GetSynchLocations(pattern_grps, d_synch_all, expected_number_patterns=2, isinit=1)
        
        # print(found_pattern_startend_idx)
        
        df_power_nilm, fig = self.GetROISection(df_power_log, found_pattern_startend_idx[0][1]-(d_synch_all-3*d_synch), found_pattern_startend_idx[-1][0]+d_synch_all-3*d_synch,  isplot=1, label= 'NILM Consumption') 
        
        return df_power_nilm, fig
            
    def RepetitionROIExtraction(self, df_power_nilm, d_synch_all, d_synch, num_repeated_expt):
        
        var_name = 'expId'
        print('Mark repeated experiment sections...')
        
        df_power_nilm_exprmnts = self.ExtractROISections(df_power_nilm.copy(), var_name=var_name, expected_number_patterns=num_repeated_expt, d_synch_all=d_synch_all, d_synch=d_synch)
        
        num_extracted_sections = len(df_power_nilm_exprmnts[var_name].unique()) 
        
        extracted_roi_durations = df_power_nilm_exprmnts[var_name].value_counts()
        self._CheckNumExtractedSections(var_name, num_repeated_expt, num_extracted_sections, extracted_roi_durations)
        
        return df_power_nilm_exprmnts
    
    def ScaledLoadsROIExtraction(self, df_power_nilm_exprmnts, d_synch_all, d_synch, num_Nh_chunks):
        
        dfs = []
        var_name='NhId'
        expIds = list(df_power_nilm_exprmnts["expId"].unique())
        
        for expId in util.tqdm(expIds, ncols=100):
            # print('expId: ', expId)
            sourceId = {'expId': expId}
            df_power_nilm_exprmnt = df_power_nilm_exprmnts[df_power_nilm_exprmnts["expId"]==expId]
            
            df_Nh_chunk = self.ExtractROISections(df_power_nilm_exprmnt.copy(), var_name=var_name, 
                expected_number_patterns=num_Nh_chunks, d_synch_all=d_synch_all, d_synch=d_synch)
            
            num_extracted_sections = len(df_Nh_chunk[var_name].unique()) 
            
            extracted_roi_durations = df_Nh_chunk[var_name].value_counts()
            self._CheckNumExtractedSections(var_name, num_Nh_chunks, num_extracted_sections, extracted_roi_durations, sourceId)

            dfs.append(df_Nh_chunk)
        df_power_nilm_Nh_chunk = pd.concat(dfs, axis=0, ignore_index=False) 
        
        return df_power_nilm_Nh_chunk  
   
    def MultiProcessesROIExtraction(self, df_power_nilm_Nh_chunk, d_synch_all, d_synch, num_Nps):
        
        dfs = []
        var_name='NpId'
        expIds = list(df_power_nilm_Nh_chunk["expId"].unique())
        
        for expId in util.tqdm(expIds, ncols=100):
            # print('expId: ', expId)
            df_wu_agg = df_power_nilm_Nh_chunk[df_power_nilm_Nh_chunk["expId"]==expId]
            NhIds = list(df_wu_agg["NhId"].unique())
            
            for NhId in NhIds:
                # print('NhId: ', NhId)
                sourceId = {'expId': expId, 'NhId': NhId}
                df_wu_agg_sec_1 = df_wu_agg[df_wu_agg["NhId"]==NhId]
                df_Np = self.ExtractROISections(df_wu_agg_sec_1.copy(), var_name=var_name, expected_number_patterns=num_Nps, d_synch_all=d_synch_all, d_synch=d_synch)
                num_extracted_sections = len(df_Np[var_name].unique()) 
                
                extracted_roi_durations = df_Np[var_name].value_counts()
                self._CheckNumExtractedSections(var_name, num_Nps, num_extracted_sections, extracted_roi_durations, sourceId)              
                    
                dfs.append(df_Np)
                
        df_power_nilm_Nps = pd.concat(dfs, axis=0, ignore_index=False)  
         
        return df_power_nilm_Nps
    
    def OnlineLoadROIExtraction(self, df_power_nilm_Nps, d_synch_all, d_synch, num_online_chunks):
        
        dfs = []
        var_name='onlineId'
        expIds = list(df_power_nilm_Nps["expId"].unique())
        
        for expId in util.tqdm(expIds, ncols=100):
            # print('expId: ', expId)
            df_wu_agg = df_power_nilm_Nps[df_power_nilm_Nps["expId"]==expId]
            NhIds = list(df_wu_agg["NhId"].unique())
            
            for NhId in NhIds:
                # print('NhId: ', NhId)
                df_wu_agg_sec_1 = df_wu_agg[df_wu_agg["NhId"]==NhId]
                NpIds = list(df_wu_agg_sec_1["NpId"].unique())
                
                for NpId in NpIds:
                    # print('NpId: ', NpId)
                    sourceId = {'expId': expId, 'NhId':NhId, 'NpId':NpId}
                    df_Np = df_wu_agg_sec_1[(df_wu_agg_sec_1["NpId"]==NpId)]
                    df_online = self.ExtractROISections(df_power_agg=df_Np.copy(), var_name=var_name, expected_number_patterns=num_online_chunks, d_synch_all=d_synch_all, d_synch=d_synch)
                    num_extracted_sections = len(df_online[var_name].unique()) 
                    
                    extracted_roi_durations = df_online[var_name].value_counts()
                    self._CheckNumExtractedSections(var_name, num_online_chunks, num_extracted_sections, extracted_roi_durations, sourceId)
            
                    dfs.append(df_online)
                    
        df_power_nilm_online = pd.concat(dfs, axis=0, ignore_index=False)  

        df_power_nilm_online = self._CleanNILMROI(df_power_nilm_online)

        return df_power_nilm_online
    
    def DiscardCorruptedExpriments(self, df_power_nilm_Nps_online):
        
        self.corrupted_expIds = list(set(self.corrupted_expIds))
        print("discarding corrupted expIds; can be retrieved from errorlog file: ", self.corrupted_expIds)

        df_power_nilm_Nps_online = df_power_nilm_Nps_online.loc[lambda df: ~df["expId"].isin(self.corrupted_expIds)]
        
        # df_power_nilm_Nps_online = self._CleanNILMROI(df_power_nilm_Nps_online)
        
        return df_power_nilm_Nps_online
    
    def PlotSelections(self, df_main, var_name, figure_path=None, figname=""):
        
        fig, ax = plt.subplots(figsize = [FIGURE_WIDTH, FIGURE_HEIGHT], nrows=1, ncols=1, constrained_layout=True)
        util.PlotROIs(ax, df_main, var_name)

        if figure_path:
            util.SaveFigure("{}/meter_log_disagg_{}".format(figure_path, figname), fig)
        
def main(result_path):
    """
    Extracts the region of interest (ROI) sections of the target active algorithm hierarchicaly and recursively from system-level power readings 
    """
    
    print('MeterROIExtraction...')
    
    # %% create debug figure directory 
    figure_path = "{}/{}/{}".format(result_path, setting_dict["meter_log"]["log_dir"], setting_dict["meter_log"]["debug_figure_dir"])
    util.CreateDir(figure_path)
        
    # %%
    ObjSeqPattern = AlgCostSignalExtractor(result_path)
    ObjSeqPattern.ResetErrorLog()
    
    # %%
    print('Loading experiment settings...')
    ObjSeqPattern.LoadExprimentSetting()
    exprmnt_setting_dict = ObjSeqPattern.exprmnt_setting_dict
    util.PrintDict(exprmnt_setting_dict)

    isdays = exprmnt_setting_dict['isdays']
    disaggtimewindow = exprmnt_setting_dict['disaggtimewindow']
    num_repeated_expt = exprmnt_setting_dict['num_repeated_expt']
    num_Nh_chunks = exprmnt_setting_dict['num_Nh_chunks']
    num_Nps = exprmnt_setting_dict['num_Nps']
    Nh_chunks = exprmnt_setting_dict['Nh_chunks']
    Nps = exprmnt_setting_dict['Nps']
    init_d = exprmnt_setting_dict['init_d']
    exprmnt_d = exprmnt_setting_dict['exprmnt_d']
    Nhchunk_d = exprmnt_setting_dict['Nhchunk_d']
    Np_d = exprmnt_setting_dict['Np_d']
    onlinedw_d = exprmnt_setting_dict['onlinedw_d']
    synch_len = onlinedw_d
    num_online_chunks = 1
     
    # %%
    print('Loading wattsup smart meter reading...')
    df_power_log, fig = ObjSeqPattern.LoadMeterPowerReading()
    util.SaveFigure("{}/meter_log_bg_removed".format(figure_path), fig)

    # %%
    print('Extracting NILMROI...')
    d_synch_all = init_d + exprmnt_d + Nhchunk_d + Np_d + onlinedw_d
    
    try:
        df_power_nilm, fig = ObjSeqPattern.NILMAlgROIExtraction(df_power_log.copy(), 5*d_synch_all, init_d)
    except:
        raise 'NILM section extraction error!'
    finally:
        util.SaveFigure("{}/meter_log_alg".format(figure_path), fig)
       
    # %%
    print('Extracting RepetitionROIs...')
    d_synch_all = d_synch_all - init_d
    df_power_nilm_exprmnts = ObjSeqPattern.RepetitionROIExtraction(df_power_nilm.copy(), 2*5*d_synch_all, exprmnt_d, num_repeated_expt)
    ObjSeqPattern.PlotSelections(df_power_nilm_exprmnts, 'expId', figure_path = figure_path, figname = 'all_experiments')
    
    # %%
    print('Extracting LoadscaleROIs...')
    d_synch_all = d_synch_all - exprmnt_d
    df_power_nilm_Nh_chunks = ObjSeqPattern.ScaledLoadsROIExtraction(df_power_nilm_exprmnts.copy(), 2*5*d_synch_all, Nhchunk_d, num_Nh_chunks)
    expIds = list(df_power_nilm_Nh_chunks["expId"].unique())
    
    for expId in util.tqdm(expIds, ncols=40):
        # print('plotting expId: ', expId)
        df = df_power_nilm_Nh_chunks[(df_power_nilm_Nh_chunks["expId"]==expId)]
        ObjSeqPattern.PlotSelections(df, 'NhId', figure_path = figure_path, figname = "expId_{}".format(expId))
    
    # %%
    print('Extracting MultiprocessROIs...')
    d_synch_all = d_synch_all - Nhchunk_d
    df_power_nilm_Nps = ObjSeqPattern.MultiProcessesROIExtraction(df_power_nilm_Nh_chunks, 2*5*d_synch_all, Np_d, num_Nps)
    expIds = list(df_power_nilm_Nps["expId"].unique())
    
    for expId in util.tqdm(expIds, ncols=100):
        # print('plotting expId: ', expId)
        df_exprmnt = df_power_nilm_Nps[(df_power_nilm_Nps["expId"]==expId)]
        NhIds = list(df_exprmnt["NhId"].unique())
        
        for NhId in NhIds:
            # print('plotting NhId: ', NhId)
            df = df_exprmnt[(df_exprmnt["NhId"]==NhId)]
            ObjSeqPattern.PlotSelections(df, 'NpId', figure_path = figure_path, figname = "expId_{}_NhId_{}".format(expId, Nh_chunks[NhId]))

    # %%
    print('Extracting consumptions of the algorithm...')
    d_synch_all = d_synch_all - onlinedw_d
    df_power_nilm_Nps_online = ObjSeqPattern.OnlineLoadROIExtraction(df_power_nilm_Nps.copy(), d_synch_all, onlinedw_d, num_online_chunks)
    ObjSeqPattern.SaveExtractedData(df_power_nilm_Nps_online.copy(), iscleaned = 0)
    expIds = list(df_power_nilm_Nps_online["expId"].unique())
    
    for expId in util.tqdm(expIds, ncols=100):
        # print('plotting expId: ', expId)
        df_exprmnt = df_power_nilm_Nps_online[(df_power_nilm_Nps_online["expId"]==expId)]
        NhIds = list(df_exprmnt["NhId"].unique())
        
        for NhId in NhIds:
            # print('plotting NhId: ', NhId)
            df = df_exprmnt[(df_exprmnt["NhId"]==NhId)]
            ObjSeqPattern.PlotSelections(df, 'NpId', figure_path = figure_path, figname = "expId_{}_NhId_{}_alg".format(expId, Nh_chunks[NhId]))

    # %%
    print('Discarding corrupted experiment sections...')
    # corrupted_expIds = ObjSeqPattern.corrupted_expIds
    df_power_nilm_Nps_online = ObjSeqPattern.DiscardCorruptedExpriments(df_power_nilm_Nps_online)
    
    # %%
    print('Saving extracted and disaggregated data...')
    ObjSeqPattern.SaveExtractedData(df_power_nilm_Nps_online)
    # %%
    print('Done!')

if __name__ == '__main__': 
    
    util.clear()
    
    # selecting target system for monitored logs working directory
    system_name, result_path = util.GetTargetDir(result_path, "target system")
    print("\nResult directory: " + result_path)
    
    # select cost log directory
    result_folder, result_path = util.GetTargetDir(result_path, "target log folder")
    print("Result directory is: " + result_path)

    # hierarchical and recursive ROI extraction from system-level power measurement
    main(result_path)

    msg = "{}\n The result of the ROI extraction can be found in {}/ and \n illustration of the ROIs at different levels are placed in {}/.\n{}"\
            .format("*"*60, setting_dict["meter_log"]["log_dir"], setting_dict["meter_log"]["debug_figure_dir"], "*"*60)
    print(msg)
    
    ## %% cleaned roi visualization
    # ROIPlotter.main(result_path)