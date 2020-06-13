# -*- coding: utf-8 -*-
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
from CostModeler import CostModeler 

base_path = os.path.dirname(os.getcwd())
code_path, data_path, main_result_path = util.GetWorkingDirs(base_path) 
sys.path.append(base_path)
sys.path.append(code_path)

setting_dict  = util.LoadJson("{}/setting.json".format(src_path))

#%% plot settings
FIGURE_HEIGHT = 3
FIGURE_WIDTH = 4.5
FIGURE_LBL_FONTSIZE_MAIN = 20

util.SetFigureStyle(FIGURE_WIDTH, FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN)
kws_online_2 = util.kws_online_2
kws_box_2 = util.kws_box_2
linestyles = util.linestyles
filled_markers = util.filled_markers

units = {'ProcTime_FromSystem': ' (sec)', 'ProcTime_FromMeter': ' (sec)', 'ProcEnergy_FromMeter': ' (wh)'}

# %%
class CostAnalyzer(CostModeler):
    """
    Cost analysis module to evaluate the computation cost of the NILM algorithm. 
    It renders multiple analytical and correlation plots to illustrate the associated processing time and energy costs of the NILM algorithm 
    corresponding to the various experiment configurations. 
    Cost prediction using the cost models for given load scale and multiprocessing settings, 
    determination of system load scale capacity at disaggregation time-window constraint, 
    and more importantly, evaluation of NILM algorithm's energy costs as compared to house energy loads or 
    expected energy efficiency gain from using the algorithm.
    """
        
    def __init__(self, data_path, system_name):

        super(CostAnalyzer, self).__init__(data_path, system_name)
        
        self.data_path = data_path 

        self.result_path = "{}/{}".format(self.data_path, setting_dict["analysis"]["results_dir"])
        util.CreateDir(self.result_path)
        
        self.figure_path = "{}/{}".format(self.result_path , setting_dict["analysis"]["figure_dir"])
        util.CreateDir(self.figure_path)

        self.costreport_filename = "{}/{}".format(self.result_path , setting_dict["analysis"]["cost_pred_report"])
        self.costreport_meta_filename = "{}/{}".format(self.result_path , setting_dict["analysis"]["report_meta"])
    
    def LoadWorkingDataset(self):
        self.dataset = pd.read_csv(self.filename_mldataset)
        
    def FeatureAnalysis(self, issave=True, isfigshow=False):
        """Feature analysis such as illustrations of the monitored costs, correlations and comparison of system and meter measurements."""
         
        print('monitored cost dataset analysis...')
        label_fontsize = FIGURE_LBL_FONTSIZE_MAIN 

        dataset = self.dataset.copy()
        dataset.sort_values(by=['Number of Houses', 'Number of Processes'], ascending=True, inplace=True)

        dataset["ProcTime (" +r'$\Delta$'+ ")" + units['ProcTime_FromMeter']] = dataset['ProcTime_FromMeter'] - dataset['ProcTime_FromSystem']
        dataset["ProcTime (" +r'$\Delta$' + ") (%)"] = 100*dataset["ProcTime (" +r'$\Delta$'+ ")" + units['ProcTime_FromMeter']].divide(dataset["ProcTime_FromMeter"], axis="index").copy()
        
        if "ProcTime_FromSystem" in self.targets:
            self.targets.remove("ProcTime_FromSystem")
        
        # Correlation analysis
        label_vars = self.features[:]
        label_vars.extend(self.targets)
        train_data = dataset[label_vars]
        cm = train_data.corr()
        mask = np.zeros_like(cm, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        fig, ax = plt.subplots(figsize=(6, 6))
        fmt = '.2f'
        label_vars = [var+' (target)' if var in self.targets else var+' (feature)'  for var in label_vars]
        annot_kws = {"size":label_fontsize+2, "ha":'center',"va":'top'}
        sns.heatmap(cm, annot=True, annot_kws=annot_kws, fmt=fmt, square=True, cmap='coolwarm', mask=mask, ax=ax, linewidths=0.1)
        plt.xticks(rotation=60, fontsize=label_fontsize)
        plt.yticks(rotation=0, fontsize=label_fontsize)
        ax.set_xticklabels(label_vars, fontsize=label_fontsize)
        ax.set_yticklabels(label_vars, fontsize=label_fontsize)
        ax.set_xticks(np.arange(0, len(label_vars), 1))
        ax.set_yticks(np.arange(0.5, len(label_vars), 1))
        ax.tick_params(axis='both', which='major', labelsize=label_fontsize)   
        filename_plot = "{}/correlation_analysis".format(self.figure_path)
        util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)

        # %% Feature analysis
        dataset['Number of Houses'] = dataset['Number of Houses'].astype('int')
        dataset['Number of Processes'] = dataset['Number of Processes'].astype('int')
       
        y_vars = ["ProcTime (" +r'$\Delta$'+ ")" + units['ProcTime_FromMeter'], "ProcTime (" +r'$\Delta$' + ") (%)"]
        x_var = 'Number of Processes'
        # print(dataset[y_vars + [x_var]].groupby([x_var]).describe())
        for i, y_var in enumerate(y_vars):
            fig, ax = plt.subplots()
            fig = sns.catplot(x=x_var, y=y_var, kind="box", data=dataset, **kws_box_2)
            plt.xlabel(x_var, fontsize=label_fontsize)
            plt.ylabel(y_var, fontsize=label_fontsize) 
            y_var2 = y_var        
            if '%' in y_var: 
                y_var2 = y_var.replace("(%)", "_perc")
            y_var2 =  y_var2.replace("$", "").replace("\\", "").replace("(", "").replace(")", "")
           
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            X, Y = dataset[x_var].values, dataset[y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}".format(self.figure_path, x_var, y_var2)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)

        y_vars = self.targets[:]
        x_var = 'Number of Processes'
        hue ='Number of Houses'        
        for i, y_var in enumerate(y_vars):
            fig, ax = plt.subplots()
            fig = sns.catplot(kind="point", x=x_var, y=y_var, data=dataset,  hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2) 
            # plt.legend(fontsize=label_fontsize - 1, frameon=True, framealpha=0.5, title=hue, ncol=2, loc='upper right', bbox_to_anchor=(1, 0.99))
            plt.xlabel(x_var, fontsize=label_fontsize)  
            plt.ylabel(y_var + units[y_var], fontsize=label_fontsize)   
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize) 
            X, Y = dataset[x_var].values, dataset[y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)

            dataset[y_var+"_puh"] = dataset[y_var].divide(dataset["Number of Houses"], axis="index").copy()
            fig = sns.catplot(kind="point", x=x_var, y=y_var+"_puh", data=dataset, hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10,   legend=False, **kws_online_2)
            plt.xlabel(x_var, fontsize=label_fontsize)  
            plt.ylabel(y_var + units[y_var], fontsize=label_fontsize)   
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=2, loc='upper right')
            X, Y = dataset[x_var].values, dataset[y_var+"_puh"].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}_puh".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)
 
        # print(dataset[[y_var +"_puh" for y_var in y_vars] + [x_var]].groupby([x_var]).describe()) 
        y_vars = ['house_energy_mean_kwh', "house_energy_median_kwh", 'house_energy_max_kwh']
        x_var = 'Number of Processes'
        hue ='Number of Houses'     
        for i, y_var in enumerate(y_vars):
            fig = sns.catplot(kind="point", x=x_var, y=y_var, data=dataset,  hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2) 
            plt.xlabel(x_var, fontsize=label_fontsize)  
            plt.ylabel(y_var, fontsize=label_fontsize)   
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize) 
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor = (1., 1+0.21*2))
            X, Y = dataset[x_var].values, dataset[y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)
        # print(dataset[y_vars + [x_var]].groupby([x_var]).describe()) 

        y_vars = ['house_energy_total_kwh']
        x_var = 'Number of Processes'
        hue ='Number of Houses'     
        for i, y_var in enumerate(y_vars):
            dataset[y_var+"_puh"] = dataset[y_var].divide(dataset["Number of Houses"], axis="index").copy()
            fig = sns.catplot(kind="point", x=x_var, y=y_var+"_puh", data=dataset, hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10,   legend=False, **kws_online_2)
            plt.xlabel(x_var, fontsize=label_fontsize)  
            plt.ylabel(y_var, fontsize=label_fontsize)   
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor = (1., 1+0.21*2))
            X, Y = dataset[x_var].values, dataset[y_var+"_puh"].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}_puh".format(self.figure_path, x_var, y_var)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow, issave=issave)      

    def PredictionSystem(self, lst_numProcesses, lst_numHouses, targets, model_name="RF"):
        
        print('Prediction System Computational Costs')

        Nh_Np_dict = dict()
        for Nh in lst_numHouses:
            Nh_Np_dict[Nh] = lst_numProcesses
        df_pred = pd.DataFrame.from_dict(Nh_Np_dict)
        df_pred = df_pred.melt(value_name='Number of Processes', var_name='Number of Houses')
        
        for target in targets:
            print("target: ", target)
            df_pred[target] = self.PredictCost(df_pred, target, model_name)
            df_pred[target] = df_pred[target].multiply(df_pred["Number of Houses"], axis="index")
        
        print('Done!')
        
        return df_pred
        
        print('Prediction System Computational Costs')
        # prepare pred dataframe
        
        Nh_Np_dict = dict()
        for Nh in lst_numHouses:
            Nh_Np_dict[Nh] = lst_numProcesses
        df_pred = pd.DataFrame.from_dict(Nh_Np_dict)
        df_pred = df_pred.melt(value_name='Number of Processes', var_name='Number of Houses')
        # print(df_pred)
        
        for target in targets:
            print("target: ", target)
            df_pred[target] = self.MakePrediction(df_pred, target, model_name)
            df_pred[target] = df_pred[target].multiply(df_pred["Number of Houses"], axis="index")
        
        print('Done!')
        
        return df_pred
    
    def SystemAnalysis(self, df_pred, bg_power, pred_variables, disaggwindow_min):
        """Provides daily, monthly and yearly energy cost predictions."""
        
        print('System Performance Analysis')
        dw_perday_slots = 24*60/disaggwindow_min
        dw_permonth_slots = 30*dw_perday_slots
        dw_peryear_slots = 12*dw_permonth_slots
        print([dw_perday_slots, dw_permonth_slots, dw_peryear_slots])
        
        new_pred_variables =[]
        pred_variables_update = []
        
        for var in pred_variables:
            new_vars = [var+'-perday', var+'-permonth', var+'-peryear']
            df_pred[new_vars[0]] = df_pred[var]*dw_perday_slots
            df_pred[new_vars[1]] = df_pred[var]*dw_permonth_slots
            df_pred[new_vars[2]] = df_pred[var]*dw_peryear_slots
            new_pred_variables.extend(new_vars)
            df_pred.rename(columns={var:var+'_perdw'},  inplace=True)
            pred_variables_update.append(var+'_perdw')
        
        # print(df_pred.columns)
        new_pred_variables.extend(new_pred_variables)   
        print(pred_variables_update)
        # actual_cols = [var in pred_variables if not var.contains('pred')]
        fix_variables = ['Number of Houses', 'Number of Processes']
        all_variables = fix_variables[:]
        all_variables.extend(new_pred_variables)
        # print(all_variables)
        df = df_pred[all_variables].melt(fix_variables, var_name='pred_variables',  value_name='values')
        
        analysis_cols = df['pred_variables'].str.split("-", n = 1, expand = True)
        df['pred_variables'] = analysis_cols[0]
        df['pred_windows'] = analysis_cols[1]
        
        pred_windows = list(df['pred_windows'].unique())
        x_var = 'Number of Processes'
        print("\nNILM SYSTEM COMPUTATIONAL COST PREDICTION\n")
        
        for pred_window in pred_windows:
            print(pred_window)
            sns.catplot(kind="point", x=x_var, y="values", col= 'pred_variables',  hue='Number of Houses', data=df[df.pred_windows==pred_window])
            filename_plot = "{}/{}_vs_num_houses_system_analysis_model_{}_perd.pdf".format(self.result_path, x_var.replace(" ", "_").lower(), pred_window)
            print(filename_plot)
            plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
            plt.show()

        for y_var in pred_variables_update:
            splot = sns.catplot(kind="point", x=x_var, y=y_var, data=df_pred, hue='Number of Houses',  **kws_online_2)
            filename_plot = "{}/{}_vs_{}_vs_num_houses_system_analysis_model_perd.pdf".format(self.result_path, x_var.replace(" ", "_").lower(), y_var.replace(" ", "_").lower(), pred_window)
            print(filename_plot)
            plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
            plt.show()
        
        print('Done!')  

    def CapacityAnalysis(self):
        """Generates summary table for cost analysis for comparsion of NILM algorithm cost verses average house loads at perunit house, and system load scale capacity."""
        
        print('cost prediction analysis report...')

        targets = ["ProcTime_FromMeter", "ProcEnergy_FromMeter"]
        model_name="ME"

        models_PT, _ = self.LoadTrainedModel(targets[0])
        MEModel_PT = models_PT['models'][model_name]
        models_PE, _ = self.LoadTrainedModel(targets[1])
        MEModel_PE = models_PE['models'][model_name]

        dict_cost_analysis = {
            "Np": sorted(list(self.dataset["Number of Processes"].unique())),
            "AHL (kwh)": self.dataset["house_energy_median_kwh"].mean(),
            "PT_puh (sec)": MEModel_PT[(targets[0], "mean")].values,
            "PE_puh (wh)": MEModel_PE[(targets[1], "mean")].values,
            "DW (minutes)": self.dataset["dw_minutes"].max(),
            "SE (wh)": self.dataset["meter_energy_bg_dw_wh"].mean()
        }
        df_cost_analysis = pd.DataFrame.from_dict(dict_cost_analysis)
        df_cost_analysis["PE_puh/AHL (%)"] = 100*df_cost_analysis["PE_puh (wh)"].divide(1000*df_cost_analysis["AHL (kwh)"])
        df_cost_analysis["Nh_max"] = 60*df_cost_analysis["DW (minutes)"].divide(df_cost_analysis["PT_puh (sec)"]).astype(int)
        df_cost_analysis["THL (kwh)"] = df_cost_analysis["AHL (kwh)"].multiply(df_cost_analysis["Nh_max"])
        df_cost_analysis["PE (wh)"] = df_cost_analysis["PE_puh (wh)"].multiply(df_cost_analysis["Nh_max"])
        df_cost_analysis["PE_total (wh)"] = df_cost_analysis["PE (wh)"].add(df_cost_analysis["SE (wh)"])
        df_cost_analysis["PE_total/THL (%)"] = 100*df_cost_analysis["PE_total (wh)"].divide(1000*df_cost_analysis["THL (kwh)"])
        
        self.costreport_meta = { 
            "Np" : "Number of Parallel Processes in Multiprocessing",
            "PT" : "Processing Time Cost of the NILM Algorthm",
            "PE" :  "Processing Energy Cost of the NILM Algorthm",
            "puh" : "Per Unit House",
            "DW" : "Disaggregation Time-Window Size of the NILM Algorithm: In this case {} minutes".format(dict_cost_analysis["DW (minutes)"]),
            "AHL" : "Average House Load in DW", 
            "Nh_max" : "Maximum Load Scale or Number of Houses that can be processed with in DW", 
            "THL" : "Total House Load in DW from Nh_max Houses", 
            "SE" : "Static or background energy consumption of the system or server within DW"
        }

        self.df_cost_analysis = df_cost_analysis.copy()

        # print(self.df_cost_analysis)
        util.PrintDict(self.costreport_meta)

        util.SaveDatatoCSV(self.costreport_filename, self.df_cost_analysis)
        util.SaveJson(self.costreport_meta_filename, self.costreport_meta)

        return df_cost_analysis

@util.timer
def main():
    
    util.clear()

    # selecting target cost dataset working directory
    system_name, data_path = util.GetTargetDir(main_result_path, "server name")
    print("Target directory is: " + data_path)

    # initialize cost analysis module
    ObjAnalyzer = CostAnalyzer(data_path=data_path, system_name=system_name)

    # load working cost dataset
    ObjAnalyzer.LoadWorkingDataset()

    # plots associated costs, correlations and comparisons
    ObjAnalyzer.FeatureAnalysis(issave=True, isfigshow=False)

    # generate predicted energy costs on daily, monthly and yearly bases with breakdown for the NILM algorithm and static power costs
    # ObjAnalyzer.PredictionSystem()

    # generate summary table of cost analysis of the NILM algorithm
    df_cost_analysis = ObjAnalyzer.CapacityAnalysis()
    print(df_cost_analysis)
  
if __name__ == '__main__': 

    main()

    msg = "{}\n The cost analysis and prediction results can be found in /{}.\n{}"\
        .format("*"*60, setting_dict["analysis"]["results_dir"], "*"*60)

    print(msg) 