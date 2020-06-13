"""
Created on Fri Aug 30 17:33:21 2019

@author: Mulugeta W.Asres

"""
import warnings
warnings.filterwarnings("ignore")

import sys, os, gc
import numpy as np, pandas as pd
import time

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

from MeterReadingROIExtractor import AlgCostSignalExtractor

#%% plot settings
FIGURE_WIDTH = 6.4
FIGURE_HEIGHT = 2
FIGURE_LBL_FONTSIZE_MAIN = 10

util.SetFigureStyle(FIGURE_WIDTH, FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN)

# %%
def remove_unused_categories(df):
    
    cat_cols = df.select_dtypes('category').columns.tolist()
    for c in cat_cols:
        df[c].cat.remove_unused_categories(inplace=True)

def ReconstructSections(df_power_log, exprmnt_setting_dict, filename_meter_log_roi_uncleaned): # after removal of corrupted ROI sections
    
    df_wupower_extracted = pd.read_csv(filename_meter_log_roi_uncleaned)
    df_wupower_extracted.rename(columns={'P_w':'Watts', 'P_bg_w':'Watts_bg', 'NhId':'Nh', 'NpId':'Np'}, inplace=True)
    df_wupower_extracted['Nh'] = df_wupower_extracted['Nh'].apply(lambda x: exprmnt_setting_dict['Nh_chunks'][x])
    df_wupower_extracted['expId'] = df_wupower_extracted['expId'] + 1
    
    df_wupower_extracted.set_index('t', inplace=True)
    df_power_log.set_index('t', inplace=True)

    synch_span =  5*(exprmnt_setting_dict['Nhchunk_d'] + exprmnt_setting_dict['Np_d'] + exprmnt_setting_dict['onlinedw_d']) 

    # df_power_log = df_power_log.drop(df_wupower_extracted[df_wupower_extracted['Nh'].isin(setting_dict["modeling"]["exclude_data"]["num_houses"])].index) 
    skip_idx_sizes = []
    expIds = list(df_wupower_extracted['expId'].unique())
    
    try:
        for Nh_skip in setting_dict["modeling"]["exclude_data"]["num_houses"]: 
            skip_idx_sizes.append(df_wupower_extracted[df_wupower_extracted['Nh'] == Nh_skip].groupby('expId').agg({'expId':[np.size]})[('expId', 'size')].min() + 2*synch_span)
            df_skip = df_wupower_extracted[df_wupower_extracted['Nh'] == Nh_skip]
            if not df_skip.empty:
                for expId in expIds:
                    skip_idx = df_skip[df_skip['expId'] == expId].index.values
                    df_power_log.drop(df_skip.loc[skip_idx[0]:skip_idx[-1], :].index, inplace=True) 
    except Exception as ex:
        print("ReconstructSections Error: {}".format(ex))

    df_power_log.reset_index(drop=False, inplace=True)

    if len(skip_idx_sizes) > 1:
        t_steps = df_power_log['t'].diff()
        t_steps = t_steps.iloc[1:]
        t_gaps = t_steps[t_steps >= np.min(skip_idx_sizes)] 
        for i in range(len(t_gaps)):
            start_idx = t_gaps.index[i]
            # print((start_idx,  t_gaps.values[i], df_power_log.index[-1]))
            df_power_log['t'].iloc[start_idx:] = df_power_log['t'].iloc[start_idx:].apply(lambda x: x - t_gaps.values[i] + 1)

    df_wupower_extracted.reset_index(drop=False, inplace=True)
    expIds = list(df_wupower_extracted.expId.unique())
    expIds_exclude = []

    for expId in expIds:
        df_exprmnt = df_wupower_extracted[(df_wupower_extracted.expId==expId)]
        Nhs = list(df_exprmnt.Nh.unique())
        for Nh_skip in setting_dict["modeling"]["exclude_data"]["num_houses"]:
            df = df_exprmnt[(df_exprmnt.Nh==Nh_skip)]
            if not df.empty:
                Nhs_sel = [Nh for Nh in Nhs if Nh > Nh_skip]
                # print((Nh_skip, Nhs_sel))
                df_wupower_extracted[~df_wupower_extracted.expId.isin(expIds_exclude)].loc[df_wupower_extracted.Nh.isin(Nhs_sel), 't'] = \
                        df_wupower_extracted[~df_wupower_extracted.expId.isin(expIds_exclude)].loc[df_wupower_extracted.Nh.isin(Nhs_sel), 't'] - df.shape[0] - 2*synch_span
        
        expIds_exclude.append(expId)
    
    df_wupower_extracted = df_wupower_extracted[~df_wupower_extracted.Nh.isin(setting_dict["modeling"]["exclude_data"]["num_houses"])]
    remove_unused_categories(df_wupower_extracted)
    
    return df_wupower_extracted

@util.timer
def main(result_path):
    """
    Provides visualization tools to plot the cleaned ROIs extracted by the MeterReadingROIExtractor. 
    It also maps the generic labels of the ROIs into their actual values using the monitoring setting from CostMonitor.
    """
    
    print('MeterROIExtractionPlotter...')
    
    # %% create figure directory 
    figure_path = "{}/{}".format(result_path, setting_dict["meter_log"]["figure_dir"])
    util.CreateDir(figure_path)
    
    # %% 
    ObjSeqPattern = AlgCostSignalExtractor(result_path)
        
    print('Loading experiment settings...')
    ObjSeqPattern.LoadExprimentSetting()
    exprmnt_setting_dict = ObjSeqPattern.exprmnt_setting_dict
    # util.PrintDict(exprmnt_setting_dict)

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
    df_power_log.head()
    df_power_log.shape
   
    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    plt.legend(ncol = 2, loc='center right',  bbox_to_anchor = (1., 1+0.1), borderaxespad=0, frameon=True )
    util.SaveFigure("{}/meter_log_bg_removed_rec".format(figure_path), fig)

    # %%
    print('Reconstructing experiments...')
    
    df_wupower_extracted = ReconstructSections(df_power_log, exprmnt_setting_dict, ObjSeqPattern.filename_meter_log_roi_uncleaned)
 
    plt.plot(df_power_log.t, df_power_log.Watts + df_power_log['Watts_bg'], label= 'Total Power')
    plt.plot(df_power_log.t, df_power_log.Watts, label= 'Dynamic Power')
    plt.xlabel('t (sec)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
    plt.ylabel('P (W)', fontsize=FIGURE_LBL_FONTSIZE_MAIN)
    util.PlotGridSpacing(df_power_log.t.values, [0, df_power_log.Watts.max() + df_power_log['Watts_bg'].max()], x_gridno=6, y_gridno=6, issnscat = True)
    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    plt.legend(ncol = 2, loc='center right',  bbox_to_anchor = (1., 1+0.1), borderaxespad=0, frameon=True )
    util.SaveFigure("{}/meter_log_bg_removed_rec".format(figure_path), fig)
    
    # %%
    print('Extracting NILMROI...')
    d_synch_all = init_d + exprmnt_d + Nhchunk_d + Np_d + onlinedw_d
    
    try:
        df_power_nilm, fig = ObjSeqPattern.NILMAlgROIExtraction(df_power_log.copy(), 5*d_synch_all, init_d)
    except:
        raise 'NILM section extraction error!'
    finally:
        plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
        # plt.legend(bbox_to_anchor = (1, 1.21))
        plt.legend(loc='center right',  bbox_to_anchor = (1., 1+0.1), borderaxespad=0, frameon=True)
        util.SaveFigure("{}/meter_log_alg_rec".format(figure_path), fig)

    # %%
    print('Extracting RepetitionROIs...')
    ObjSeqPattern.PlotSelections(df_wupower_extracted, 'expId', figure_path = figure_path, figname = 'all_experiments'+'_rec')
    
    # %%
    print('Extracting LoadscaleROIs...')
    df_wupower_extracted_cleaned = pd.read_csv(ObjSeqPattern.filename_meter_log_roi)
    df_wupower_extracted_cleaned.rename(columns={'P_w':'Watts', 'P_bg_w':'Watts_bg', 'NhId':'Nh', 'NpId':'Np'}, inplace=True)
    df_wupower_extracted_cleaned['Nh'] = df_wupower_extracted_cleaned['Nh'].apply(lambda x: exprmnt_setting_dict['Nh_chunks'][x])
    df_wupower_extracted_cleaned['expId'] = df_wupower_extracted_cleaned['expId'] + 1
    df_wupower_extracted = df_wupower_extracted[df_wupower_extracted.expId.isin(df_wupower_extracted_cleaned.expId.unique())]

    expIds = list(df_wupower_extracted.expId.unique())
    
    for expId in util.tqdm(expIds, ncols=100):
        # print('plotting expId: ', expId)
        df = df_wupower_extracted[(df_wupower_extracted.expId==expId)]
        ObjSeqPattern.PlotSelections(df, 'Nh', figure_path = figure_path, figname = "expId_{}_rec".format(expId))
    
    # %%
    print('Extracting MultiprocessROIs...')
    expIds = list(df_wupower_extracted.expId.unique())
    
    for expId in util.tqdm(expIds, ncols=100):
        # print('plotting expId: ', expId)
        df_exprmnt = df_wupower_extracted[(df_wupower_extracted.expId==expId)]
        Nhs = list(df_exprmnt.Nh.unique())
        
        for Nh in Nhs:
            # print('plotting Nhs: ', Nh)
            df = df_exprmnt[(df_exprmnt.Nh==Nh)]
            ObjSeqPattern.PlotSelections(df, 'Np', figure_path = figure_path, figname = "expId_{}_NhId_{}_rec".format(expId, Nh))
       
    print('Done!')

# %%
if __name__ == '__main__': 

    util.clear()

    # selecting target system for monitored logs working directory
    system_name, result_path = util.GetTargetDir(result_path, "target system")
 
    print("Result directory is: " + result_path)
    
    # select cost log directory
    result_folder, result_path = util.GetTargetDir(result_path, "target log folder")
    print("Result directory is: " + result_path)

    # clean and plot extracted ROIs hierarchically and recursively
    main(result_path)

    msg = "{}\n The illustration of the cleaned ROIs at different levels are placed in {}/.\n{}"\
            .format("*"*60, setting_dict["meter_log"]["figure_dir"], "*"*60)
    print(msg)
    