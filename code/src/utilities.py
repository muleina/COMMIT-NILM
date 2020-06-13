"""
Created on Mon Mar 30 17:33:21 2020

@author: muleina

Utilities module provides commonly shared functions and libraries across the different modules of the CCMT.

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, psutil, gc, json, pickle
import numpy as np, pandas as pd, itertools
import functools
import datetime, time
from scipy.stats import norm, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from statsmodels.distributions.empirical_distribution import ECDF
from platform import python_version
from tqdm import tqdm

import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt

clear = lambda: os.system('cls')

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth', 800)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 300)

#%% ploting settings
FIGURE_HEIGHT = 3
FIGURE_WIDTH = 4.5
FIGURE_LBL_FONTSIZE_MAIN = 20
FIGURE_SAVEFORMAT = ".png" # ".pdf" ".png"

# for catplot
filled_markers = ('o', 'v', 'X', '^', 's', '<', '+', '>', 'd', '8', 'p', 'h', 'H', 'D', 'P', '*')
linestyles = ('-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-')

ci = 95
kws_online_2 = dict(dodge=False, ci=ci, capsize=0.1, grid=True, lw=3, height=FIGURE_HEIGHT, aspect=FIGURE_WIDTH/FIGURE_HEIGHT, estimator= np.mean)
kws_box_2 = dict(dodge=False, ci=ci, height=FIGURE_HEIGHT, aspect=FIGURE_WIDTH/FIGURE_HEIGHT, estimator=np.mean)

#%% settings
# template for filename of system_logs
resource_log_filename_format = "{}/system_log_{}_DW_{}_Exprmnt_{}_Nh_{}_Np_{}"

# absolute percentage error confidence intervals for evaluation of predictions from cost models
eval_CIs = [1, 3, 5, 10] # absolute percentage error intervals
    
def timer(func):
    
    @functools.wraps(func)
    
    def wrapper_timer(*args, **kwargs):
        
        t_start = time.time()
        value = func(*args, **kwargs)
        print("Finished {} in {} secs"
             .format(func.__name__, time.time() - t_start)) #  491.4211 min first run,  450.930 min second run
        
        return value
    
    return wrapper_timer

def SetFigureStyle(FIGURE_WIDTH=FIGURE_WIDTH, FIGURE_HEIGHT=FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN=FIGURE_LBL_FONTSIZE_MAIN):
    global kws_online_2, kws_box_2
    kws_online_2 = dict(dodge=False, ci=ci, capsize=0.2, grid=True, lw=3, height=FIGURE_HEIGHT, aspect=FIGURE_WIDTH/FIGURE_HEIGHT, estimator= np.mean)
    kws_box_2 = dict(dodge=False, ci=ci, height=FIGURE_HEIGHT, aspect=FIGURE_WIDTH/FIGURE_HEIGHT, estimator=np.mean)

    sns.set(rc={'figure.figsize':(FIGURE_WIDTH,  FIGURE_HEIGHT)})
    label_fontsize = FIGURE_LBL_FONTSIZE_MAIN
    sns.set_style('whitegrid')
    # sns.set_style("ticks", {"xtick.major.size":label_fontsize, "ytick.major.size":label_fontsize, 'axes.spines.bottom': False, 'axes.spines.left': False, 'axes.spines.right': False, 'axes.spines.top': False,})
    plt.rcParams['figure.figsize'] = (FIGURE_WIDTH,  FIGURE_HEIGHT)
    plt.rcParams['legend.title_fontsize'] = label_fontsize - 3
    plt.rcParams['legend.fontsize'] = label_fontsize - 3
    plt.rcParams['legend.frameon'] = False
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.columnspacing"] = 0.1
    plt.rcParams["legend.borderaxespad"] = 0.05
    plt.rcParams["legend.borderpad"] = 0.05
    plt.rcParams["legend.handletextpad"] = 0.05
    plt.rcParams["axes.grid"] = True
    # plt.rcParams["axes.edgecolor"] = "b"

    # plt.style.use('classic')
    # plt.rcParams['grid.color'] = 'k'
    # plt.rcParams['grid.linestyle'] = ':'
    # plt.rcParams['grid.linewidth'] = 0.5

def PlotGridSpacing(X, Y, x_gridno=4, y_gridno=4, issnscat = True):
    """for formating figures axis and ticks"""
    X, Y = [x for x in X if (not np.isnan(x)) & (not np.isinf(x))], [y for y in Y if (not np.isnan(y)) & (not np.isinf(y))]
    if (len(X) < 1) | (len(Y) < 1):
        return
    axes = plt.gca()
    # xmin, xmax, ymin, ymax = axes.axis()
    xmin, xmax, ymin, ymax = np.min(X), np.max(X),  np.min(Y), np.max(Y)
    y_interval = (ymax - ymin)/(y_gridno-1)

    if not issnscat:
        x_interval = (xmax - xmin)/(x_gridno) 
        y_interval = (ymax - ymin)/(y_gridno)
        if x_interval != 0:
            j = 1
            x_round_precision = 0
            while x_interval*j < 1:
                j = 10*j
                x_round_precision = x_round_precision + 1
            x_interval = np.round(x_interval, x_round_precision)
            xaxis_lim = [xmin, xmax]
            # x_interval = np.round((xaxis_lim[1] - xaxis_lim[0])/x_gridno, x_round_precision)
            xaxis_lim = np.round([xaxis_lim[0] - x_interval*(0.20), xaxis_lim[1] + x_interval*(1.20)], x_round_precision)
            xticks = np.arange(np.max([xaxis_lim[0], xmin]), xaxis_lim[1], step=x_interval)
            xticks = np.round(xticks, x_round_precision)
            plt.xticks(xticks)
   
    # print((x_interval, y_interval))
    if y_interval != 0:
        j = 1
        y_round_precision = 0
        while y_interval*j < 1:
            j = 10*j
            y_round_precision = y_round_precision + 1
        y_interval = np.round(y_interval, y_round_precision)
        yaxis_lim = [ymin, ymax]
        # y_interval = np.round((yaxis_lim[1] - yaxis_lim[0])/y_gridno, y_round_precision)
        
        if yaxis_lim[0] > 0:    
            yaxis_lim = np.round([np.max([0, yaxis_lim[0] - y_interval*(0.20)]), yaxis_lim[1] + y_interval*(1.20)], y_round_precision)
        else:
            yaxis_lim = np.round([yaxis_lim[0] - y_interval*(0.20), yaxis_lim[1] + y_interval*(1.20)], y_round_precision)
        yticks = np.arange(np.max([yaxis_lim[0], ymin]), yaxis_lim[1], step=y_interval)
        yticks = np.round(yticks, y_round_precision)
        plt.yticks(yticks)

    axes.grid(True, which='major', axis='both')
    # plt.rc('grid', color='black')

def PlotROIs(ax, df, var_name):
    """plots ROIs inside the given upper ROI (var_name)"""
    if df.empty:
        return

    Ids = list(df[var_name].unique())
   
    for Id in Ids:
        df_roi = df[df[var_name]==Id]
        ax.plot(df_roi["t"], df_roi["Watts"], label=var_name + '_' + str(Id))

    plt.xlabel('t (sec)')
    plt.ylabel('P (W)')
    ax.tick_params(axis='both', which='major')   
    PlotGridSpacing(df["t"].values, [0, df["Watts"].max()], x_gridno=6, y_gridno=6, issnscat = True)
    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    nrow = int(np.ceil(len(Ids)/5.0))
    ncol = int(len(Ids)/nrow)
    plt.legend(ncol = ncol,loc='center right',  bbox_to_anchor = (1., 1+0.1*np.ceil(len(Ids)/ncol)), borderaxespad=0, frameon=True )
       
def SaveFigure(filepath, fig, isshow=False, issave=True):
    
    filepath = filepath.lower().replace(" ", "_")
    filpath = "{}{}".format(filepath, FIGURE_SAVEFORMAT)
    # print("Saving ", filpath)
    if issave:
        fig.savefig(filpath, dpi=300, bbox_inches='tight') 
        plt.close() 
    if isshow:
        plt.show(fig)

def PrintDict(dc):

    print("\n{}".format("*"*40))
    for key, value in dc.items():
        print("{}: {}".format(key, value))
    print("{}".format("*"*40))

def GenerateNewDirName(base_path):
    
    current_dirs = [int(dirname) for dirname in next(os.walk(base_path))[1] if dirname.isnumeric()] 
    last_dir = np.max(current_dirs) if len(current_dirs) > 0 else 0
    
    return "{}/{}".format(base_path, last_dir+1)

def CreateCostMonitorLogDirs(result_path, wum_log_dir, resource_monitor_log_dir):
    
    result_path = GenerateNewDirName(result_path)
    
    wum_log_path = "{}/{}".format(result_path, wum_log_dir)  
    resource_monitor_log_path = "{}/{}".format(result_path, resource_monitor_log_dir)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if not os.path.exists(wum_log_path):
        os.makedirs(wum_log_path)
        
    if not os.path.exists(resource_monitor_log_path):
        os.makedirs(resource_monitor_log_path)
        
    return wum_log_path, resource_monitor_log_path

def GetWorkingDirs(base_path, system_name = None):
    
    code_path = "{}/src".format(base_path)
    data_path = "{}/data".format(base_path)
    if system_name:
        result_path = "{}/results/{}".format(base_path, system_name)
    else:
         result_path = "{}/results".format(base_path)
    
    return code_path, data_path, result_path

def GetTargetDir(base_path, var_name):
    
    result_dirs = [name for name in next(os.walk(base_path))[1]]
    print("\n".join([str(i+1) + ':' + d for i, d in enumerate(result_dirs)]))
    
    if len(result_dirs) < 1:
        raise "no result log in directory: {}!".format(base_path)
	
    print("The following logs are found in the result directory. Select {}:".format(var_name))
    retry = 5
    while retry > 0:
        try:
            resultId = int(input('Enter the list number:'))
            if resultId == 0:
                retry = 0
                break	
            resultId = resultId - 1
            target_name = result_dirs[resultId]  
            retry = 0
        except:
            print('you inserted a wrong list number! try again (0 to exit):')
        retry-=1
        
        if retry == 0:
            raise "no selection of result log directory. user exists!"
        
    target_path = "{}/{}".format(base_path, target_name)
    
    return target_name, target_path

def CreateDir(dir_path):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
def ResetLogFile(filepath):
    
    with open(filepath, 'w+') as fhandle:
        fhandle.write("")
        fhandle.close()
            
def SaveJson(filepath, datadic):
    
    print("Saving: ", filepath)
    with open(filepath, 'w') as fhandle:
        fhandle.write(json.dumps(datadic, indent=4, sort_keys=True))
        fhandle.close()

def LoadJson(filepath):
    
    # print("filename: ", filepath)
    with open(filepath, 'r') as fhandle:
        setting_dict = json.load(fhandle)
        fhandle.close()
    # print(setting_dict)
   
    return setting_dict

def SaveDatatoCSV(filepath, df):
    
    # print('Saving ', filepath)
    df.to_csv(filepath, float_format='%6.5f', index=False)

def UpdateTimeWindowStart(v, isdwsize_indays=0, window_size=60, house_id_isfound=0):
    
    if house_id_isfound:  
        if isdwsize_indays==1:
            return [v[0] + datetime.timedelta(days = window_size), v[1]+1]
        else:
            return [v[0] + datetime.timedelta(minutes = window_size), v[1]+1]
    else: 
        return [v[0], v[1]]
    
def ConfidenceIntervals(x, x_name):
    
    x = np.round(np.abs(x), 5)
    ape_extremes = {"MIN_APE": np.min(x), "MAX_APE": np.max(x)}
    
    norm_cdf = ECDF(x) # fits continous function
    CI = [norm_cdf(eval_CIs[0]), norm_cdf(eval_CIs[1]), norm_cdf(eval_CIs[2]), norm_cdf(eval_CIs[3])]

    return CI, norm_cdf, x_name, ape_extremes

def CalcRvalue(y_measured, y_pred):
    Rvalue, _ = pearsonr(y_measured, y_pred)
    return np.round(Rvalue, 3)

def AdjRsquare(R2, n_sample, NUM_FEATURES=1):
    Adj_r2 = 1-(1-R2)*(n_sample-1)/(n_sample-NUM_FEATURES-1)
    return np.round(Adj_r2, 3)   

def mse_custom(y_true, y_pred):    
    return np.abs(np.round(mean_squared_error(y_true, y_pred), 6))

def r2_custom(y_true, y_pred):    
    return np.round(r2_score(y_true, y_pred), 5)

def max_ae(y_true, y_pred):
    return np.round(np.max(np.abs(y_true - y_pred)), 6)

def ape(y_true, y_pred):
    return np.abs(100*np.divide(y_pred - y_true, y_true))
    # return np.abs(100*np.divide(y_pred - y_true, y_pred))
    # return np.abs(np.array([100*(p-t)/p for t, p in zip(y_true, y_pred) if p!=0]))

def max_ape(y_true, y_pred):    
    return np.round(np.max(ape(y_true, y_pred)), 6)

def mape(y_true, y_pred):    
    return np.round(np.mean(ape(y_true, y_pred)), 6)

def mdape(y_true, y_pred):    
    return np.round(np.median(ape(y_true, y_pred)), 6)

if not python_version().startswith("2.7"):
    SetFigureStyle()
    
gc.enable()
clear = lambda: os.system('cls')
clear()