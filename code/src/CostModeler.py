"""
Created on Fri Aug 30 17:33:21 2019

@author: Mulugeta W.Asres

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc, pickle
import numpy as np,pandas as pd
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor

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
class PredictiveModel:
    """
    Provides interface for cost modeling, tuning and and evalution using statistics-based model (mean estimation) and machine learning (random forest).
    """
    def __init__(self, models_path, dataset=None, features=None, target="", model_name=None):
        
        self.dataset = dataset
        self.costmodels = dict({'models':{}, 'evals':{}})
        self.features = features
        self.target = target
        self.mlmodel_name = target
        self.aggregator = np.median # np.mean

        if model_name:
            self.mlmodel_name = self.mlmodel_name

        # self.normalizer = MinMaxScaler(feature_range=(-1, 1))

        self.seed = setting_dict["modeling"]["training"]["seed"]
        self.test_num_houses = setting_dict["modeling"]["validation"]["num_houses"]
        self.test_num_processes = setting_dict["modeling"]["validation"]["num_processes"]
        self.filename_mlmodel = "{}/costmodel_{}.pickle".format(models_path, self.target)
        self.filename_mlmodel_eval =  "{}/costmodel_{}_eval.JSON".format(models_path, self.target)

        self.cv_scores = dict()
        # self.mlmodel_scores = dict()
    
    def GetTestSelection(self):
        
        return self.dataset["Number of Houses"].isin(self.test_num_houses) & self.dataset["Number of Processes"].isin(self.test_num_processes)              
    
    def GetDataset(self, isME = 0):
        
        self.normalizer = MinMaxScaler(feature_range=(-1, 1))

        features = self.features[:]
        dataset = self.dataset.copy()
        
        dataset[self.target] = dataset[self.target].divide(dataset["Number of Houses"], axis="index")

        if (isME == 0) and ("num_houses_inv" not in features):
            dataset["num_houses_inv"] = dataset["Number of Houses"].apply(lambda x: 1/x)
            features.append("num_houses_inv")

        test_sel = self.GetTestSelection() 

        self.normalizer.fit(dataset.loc[~test_sel, self.target].values.reshape(-1, 1)) # fir normalized based on train data

        return dataset, features
    
    def StatBasedModel_predct(self, x, MEModel):
        
        return np.array([MEModel.loc[MEModel.index==xi].values[0, 0] for xi in x]) 
    
    def StatBasedModel(self):
        
        print('\nMean Estimator (ME) Model Fitting ...\n')
        model_name = "ME"
        test_sel = self.GetTestSelection() 
        dataset, features = self.GetDataset(isME=1)
        features.remove("Number of Houses")
        print('features: ', features)
        print('target: ', self.target)
        
        X_test = dataset.loc[test_sel, features].values
        y_test = dataset.loc[test_sel, self.target].values
        X = dataset.loc[:, features].values
        y = self.dataset[self.target].values

        features_target = features[:]
        features_target.append(self.target)
        Xy_train = dataset.loc[~test_sel, features_target]
        MEModel = Xy_train.groupby(by=features).agg([np.mean])
        y_pred = self.StatBasedModel_predct(X_test.T[0], MEModel)
        
        modeleval = dict()
        costmodel = MEModel
        modeleval['TEST_MSE_PUH'] = mean_squared_error(y_test, y_pred)   

        dataset[self.target + '_pred'] = self.StatBasedModel_predct(X.T[0], MEModel)
        self.dataset[self.target + '_pred'] = dataset[self.target + '_pred'].multiply(dataset["Number of Houses"], axis="index").copy()
        
        modeleval['TEST_MSE'] = mean_squared_error(self.dataset.loc[test_sel, self.target], self.dataset.loc[test_sel, self.target + '_pred'])
        self.costmodels[model_name] = modeleval
        
        self.costmodels['models'][model_name] = costmodel
        self.costmodels['evals'][model_name] = modeleval

        return model_name, features
       
    def MLBasedModel(self):
        
        print('\nRandomForest Model Training ...\n')
        
        model_name = 'RF'

        test_sel = self.GetTestSelection() 
        dataset, features = self.GetDataset()
        features.remove("Number of Houses")

        print('features: ', features)
        print('target: ', self.target)

        # X_train = pd.get_dummies(dataset.loc[~test_sel, features], prefix='Np', prefix_sep='_', dummy_na=False, columns=["Number of Processes"])
        X_train = dataset.loc[~test_sel, features].values
        y_train = dataset.loc[~test_sel, self.target].values
        # X_test = pd.get_dummies(dataset.loc[test_sel, features], prefix='Np', prefix_sep='_', dummy_na=False, columns=["Number of Processes"])
        X_test = dataset.loc[test_sel, features].values
        y_test = dataset.loc[test_sel, self.target].values
        
      
        # X = pd.get_dummies(dataset[features], prefix='Np', prefix_sep='_', dummy_na=False, columns=["Number of Processes"])
        X = dataset.loc[:, features]
        y = dataset[self.target]

        if 'ProcTime' in self.target:
             params = setting_dict["modeling"]["models"]["pt"]["rf"]
             seed = setting_dict["modeling"]["models"]["pt"]["seed"]
        elif 'ProcEnergy' in self.target:
             params = setting_dict["modeling"]["models"]["pe"]["rf"]   
             seed = setting_dict["modeling"]["models"]["pe"]["seed"]    

        print('train seed: ', seed)
        np.random.seed(seed)

        costmodel = RandomForestRegressor(**params)   
        # print(costmodel.get_params)

        # Train the costmodel on training data
        # costmodel = costmodel.fit(X_train, y_train)
        costmodel.fit(X_train, self.normalizer.transform(y_train.reshape(-1, 1)))

        # y_pred = costmodel.predict(X_test)
        y_pred = self.normalizer.inverse_transform(costmodel.predict(X_test).reshape(-1, len(y_test)))[0]

        modeleval = dict()
        modeleval['TEST_MSE_PUH'] = mean_squared_error(y_test, y_pred)   

        # self.mlmodel_scores['test_mse'] = mean_squared_error(y_test, y_pred)   
        # dataset[self.target + '_pred'] = costmodel.predict(X)
        dataset[self.target + '_pred'] = self.normalizer.inverse_transform(costmodel.predict(X).reshape(-1, len(y)))[0]
        self.dataset[self.target + '_pred'] = dataset[self.target + '_pred'].multiply(dataset["Number of Houses"], axis="index").copy()
        
        modeleval['TEST_MSE'] = mean_squared_error(self.dataset.loc[test_sel, self.target], self.dataset.loc[test_sel, self.target + '_pred'])

        self.costmodels['models'][model_name] = costmodel
        self.costmodels['evals'][model_name] = modeleval

        return model_name, features
    
    @util.timer
    def ParameterTuneGridCV(self, param_grid, n_jobs=-1):  
        
        print("K-fold cross-validation model hyper-parameter tuning using njobs =  {} ... ".format(n_jobs))        

        def plot_search(gs_result, scoring, parameter_axis):
            
            fig, ax = plt.subplots(figsize=(20, 5))
            plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

            plt.xlabel(parameter_axis)
            plt.ylabel("Score")

            X_axis = np.array(gs_result['param_' + parameter_axis].data, dtype=float)
            idx = np.argsort(X_axis)

            for scorer, color in zip(sorted(scoring), ['g', 'r', 'b', 'k']):
                for sample, style in (('train', '--'), ('test', '-')):
                    sample_score_mean = gs_result['mean_%s_%s' % (sample, scorer)]
                    sample_score_std = gs_result['std_%s_%s' % (sample, scorer)]
                    ax.fill_between(X_axis[idx], sample_score_mean[idx] - sample_score_std[idx],
                                    sample_score_mean[idx] + sample_score_std[idx],
                                    alpha=0.5 if sample == 'test' else 0, color=color)
                    ax.plot(X_axis[idx], sample_score_mean[idx], style, color=color,
                            alpha=1 if sample == 'test' else 0.7,
                            label="%s (%s)" % (scorer, sample))

                best_index = np.nonzero(gs_result['rank_test_%s' % scorer] == 1)[0][0]
                best_score = gs_result['mean_test_%s' % scorer][best_index]

                # Plot a dotted vertical line at the best score for that scorer marked by x
                ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                        linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

                # Annotate the best score for that scorer
                ax.annotate("%0.2f" % best_score,
                            (X_axis[best_index], best_score + 0.005))

            plt.legend(loc="best")
            plt.grid(False)
            plt.show()

        test_sel = self.GetTestSelection() 

        dataset, features = self.GetDataset()
        features.remove("Number of Houses")
        print("features: ", features)
        print("target: ", self.target)
        n_features = len(features)

        # X = pd.get_dummies(dataset.loc[~test_sel, features], prefix='Np', prefix_sep='_', dummy_na=False, columns=["Number of Processes"]).values
        X = dataset.loc[~test_sel, features].values
        y = dataset.loc[~test_sel, self.target].values
        y = np.squeeze(self.normalizer.transform(y.reshape(-1, 1)))
        # print(X, y)

        grouping_var = "Number of Houses"
        groups = dataset.loc[~test_sel, grouping_var]
        n_splits = groups.nunique()
        print("unique groups based on {}: {}".format(grouping_var, groups.unique()))
        groups = groups.values
       
        scoring = { 
                    # 'MSE':  'neg_mean_squared_error',
                    # 'R^2': 'r2',
                    # 'AE_MAX': make_scorer(util.max_ae, greater_is_better=False),
                    'MAPE':  make_scorer(util.mape, greater_is_better=False),
                    'MdAPE':  make_scorer(util.mdape, greater_is_better=False),
                    'APE_MAX': make_scorer(util.max_ape, greater_is_better=False)
                    }
        
        report_scores = ['mean_test_'+ score for score in scoring]
        rank_scores = ['rank_test_'+ score for score in scoring]
        report_scores.extend(rank_scores)
        report_scores.append('params')

        cv_repeat = 1
        for i in range(0, cv_repeat):
            seed = self.seed + i
            print("seed: ", seed)
            np.random.seed(seed)   

            reg_model = RandomForestRegressor(bootstrap=True, n_estimators=6, max_depth=6, random_state=110)
            
            # param_grid['random_state'] = [i+8285] 
            param_grid['random_state'] = np.random.choice(np.arange(0, 1000), 100, replace=False)
                
            gkf = GroupKFold(n_splits=n_splits).split(X, y, groups=groups)

            grid_search = GridSearchCV(estimator=reg_model, param_grid=param_grid, cv=gkf, scoring=scoring, refit='MdAPE',  return_train_score=True, n_jobs=n_jobs)

            grid_search.fit(X, y)

            df_grid_search = pd.DataFrame.from_dict((grid_search.cv_results_))
 
            print(df_grid_search.loc[df_grid_search[rank_scores].min(axis=1) <= 3, report_scores])
            # print('Best_model:', grid_search.best_estimator_)          
            '''
            for param in param_grid.keys():
                 plot_search(grid_search.cv_results_, scoring, param)
            '''
    
    def ModelEval(self, model_name, features, figure_path, isfigshow=False):
        
        label_fontsize = FIGURE_LBL_FONTSIZE_MAIN
        
        test_sel = self.GetTestSelection()     
        self.dataset[self.target+'_pred_error'] = self.dataset[self.target + '_pred']  - self.dataset[self.target]
        self.dataset[self.target+'_pred_error (%)'] = 100*self.dataset[self.target+'_pred_error'].divide(self.dataset[self.target], axis="index")                  

        modeleval = self.costmodels['evals'][model_name] 
        modeleval['ALL_MAPE'] = np.round(self.dataset[self.target+'_pred_error (%)'].abs().mean(), 3)
        modeleval['TRAIN_MAPE'] = np.round(self.dataset.loc[~test_sel, self.target+'_pred_error (%)'].abs().mean(), 3)
        modeleval['TEST_MAPE'] = np.round(self.dataset.loc[test_sel, self.target+'_pred_error (%)'].abs().mean(), 3)

        modeleval['ALL_RVALUE'] = util.CalcRvalue(self.dataset[self.target], self.dataset[self.target + '_pred'].values)
        modeleval['TRAIN_RVALUE'] = util.CalcRvalue(self.dataset.loc[~test_sel, self.target], self.dataset.loc[~test_sel, self.target + '_pred'].values)
        modeleval['TEST_RVALUE'] = util.CalcRvalue(self.dataset.loc[test_sel, self.target], self.dataset.loc[test_sel, self.target + '_pred'].values)

        modeleval['ALL_APE_CI'], norm_cdf_all, x_name_all, modeleval["APE_EXTREMES_ALL"] = util.ConfidenceIntervals(self.dataset[self.target+'_pred_error (%)'], 'Absolute Normalized Error (%)')
        modeleval['TRAIN_APE_CI'], norm_cdf_train, x_name_train, modeleval["APE_EXTREMES_TRAIN"] = util.ConfidenceIntervals(self.dataset.loc[~test_sel, self.target+'_pred_error (%)'], 'Absolute Normalized Error (%)')
        modeleval['TEST_APE_CI'], norm_cdf_test, x_name_test, modeleval["APE_EXTREMES_TEST"] = util.ConfidenceIntervals(self.dataset.loc[test_sel, self.target+'_pred_error (%)'], 'Absolute Normalized Error (%)')
        modeleval["APE_CI_THRESHOLDS"] = util.eval_CIs
        self.costmodels['evals'][model_name] = modeleval
        
        util.PrintDict(self.costmodels['evals'][model_name])        

        # Regression Fitting
        label_fontsize = 14
        plt.rcParams['legend.title_fontsize'] = label_fontsize - 1
        plt.rcParams['legend.fontsize'] = label_fontsize - 1
        plt.rcParams["legend.borderaxespad"] = 0.1
        plt.rcParams["legend.borderpad"] = 0.1

        font = {'color':'red','weight':'normal','size':label_fontsize}
    
        # Residual Distribution
        fig, ax = plt.subplots(figsize = [FIGURE_HEIGHT*1.8, FIGURE_HEIGHT/2], nrows=1, ncols=2, constrained_layout=True)
        
        plt.subplot(121)
        X = self.dataset.loc[~test_sel, self.target]
        Y = self.dataset.loc[~test_sel, self.target + '_pred']
        sns.regplot(X, Y, line_kws={"color": "red"})
        plt.text(X.min(), X.max(), r'$R = %0.2f$'% modeleval['TRAIN_RVALUE'], fontdict=font)
        plt.xlabel('Measured' + units[self.target], fontsize=label_fontsize)
        plt.ylabel('Predicted'+ units[self.target], fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=label_fontsize)   
        util.PlotGridSpacing(X, Y, x_gridno=4, y_gridno=4, issnscat = False)

        plt.subplot(122)
        X, Y = norm_cdf_train.x, norm_cdf_train.y
        sns.lineplot(x=X, y=Y, lw=2, label='CDF') 
        plt.xlabel(x_name_train, fontsize=label_fontsize)
        plt.ylabel('Probability', fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
        plt.ylim([0.0, 1.0])
        util.PlotGridSpacing(X, Y, x_gridno=4, y_gridno=4, issnscat = False)
        plt.ylim([0.0, 1.0])

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.6)
        filename_plot = "{}/{}_residual_ml_{}_TRAIN".format(figure_path, self.target, model_name)
        util.SaveFigure(filename_plot, fig, isshow=isfigshow)

        fig, ax = plt.subplots(figsize = [FIGURE_HEIGHT*1.8, FIGURE_HEIGHT/2], nrows=1, ncols=2, constrained_layout=True)
        
        plt.subplot(121)
        X = self.dataset.loc[test_sel, self.target]
        Y = self.dataset.loc[test_sel, self.target + '_pred']
        sns.regplot(X, Y, line_kws={"color": "red"})
        plt.text(X.min(), X.max(), r'$R = %0.2f$'% modeleval['TEST_RVALUE'], fontdict=font)
        plt.xlabel('Measured'+ units[self.target], fontsize=label_fontsize)
        plt.ylabel('Predicted'+ units[self.target], fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
        util.PlotGridSpacing(X, Y, x_gridno=4, y_gridno=4, issnscat = False)
        
        plt.subplot(122)
        X, Y = norm_cdf_test.x, norm_cdf_test.y
        sns.lineplot(x=X, y=Y, lw=2, label = 'CDF')     
        plt.xlabel(x_name_test, fontsize=label_fontsize)
        plt.ylabel('Probability', fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
        plt.ylim([0.0, 1.0])
        util.PlotGridSpacing(X, Y, x_gridno=4, y_gridno=4, issnscat = False)
        plt.ylim([0.0, 1.0])
        
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.6)
        filename_plot = "{}/{}_residual_ml_{}_TEST".format(figure_path, self.target, model_name)
        util.SaveFigure(filename_plot, fig, isshow=isfigshow)

        # Error and Prediction Plots
        util.SetFigureStyle(FIGURE_WIDTH, FIGURE_HEIGHT, FIGURE_LBL_FONTSIZE_MAIN)
        label_fontsize = FIGURE_LBL_FONTSIZE_MAIN
        kws_online_2 = util.kws_online_2
        kws_box_2 = util.kws_box_2
        linestyles = util.linestyles
        filled_markers = util.filled_markers

        y_vars = [self.target, self.target+'_pred']
        x_var = 'Number of Processes' 
        hue='Number of Houses'
        for y_var in y_vars:
            fig = sns.catplot(kind="point", x=x_var, y=y_var, data=self.dataset.loc[~test_sel,:], hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2)                
            plt.xlabel(x_var, fontsize=label_fontsize)
            plt.ylabel(self.target + units[self.target], fontsize=label_fontsize)
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor=(1.,1.4))
            X, Y = self.dataset.loc[~test_sel, x_var].values, self.dataset.loc[~test_sel, y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}_{}_TRAIN".format(figure_path, x_var, y_var, model_name)
            if not y_var.endswith("_pred"):
                filename_plot = "{}/{}_vs_{}_{}_TRAIN".format(figure_path, x_var, y_var, "actual")
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)

            fig = sns.catplot(kind="point", x=x_var, y=y_var, data=self.dataset.loc[test_sel,:], hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2)
            plt.xlabel(x_var, fontsize=label_fontsize)
            plt.ylabel(self.target + units[self.target], fontsize=label_fontsize)
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor=(1.,1.4))
            X, Y = self.dataset.loc[test_sel, x_var].values, self.dataset.loc[test_sel, y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}_{}_TEST".format(figure_path, x_var, y_var, model_name)
            if not y_var.endswith("_pred"):
                filename_plot = "{}/{}_vs_{}_{}_TEST".format(figure_path, x_var, y_var, "actual")
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)
           
        y_vars = [self.target+'_pred_error', self.target+'_pred_error (%)']
        x_var = 'Number of Processes'
        hue='Number of Houses'
        for y_var in y_vars:
            fig = sns.catplot(kind="point", x=x_var, y=y_var,  data=self.dataset.loc[~test_sel,:], hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2)
            plt.xlabel(x_var, fontsize=label_fontsize)
            y_var2 = y_var
            if '%' in y_var: 
                plt.ylabel("Error (%)", fontsize=label_fontsize)
                y_var2 = y_var.replace("(%)", "perc")
            else:
                plt.ylabel("Error" + units[self.target], fontsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor=(1.,1.4))
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            X, Y = self.dataset.loc[~test_sel, x_var].values, self.dataset.loc[~test_sel, y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)                
            filename_plot = "{}/{}_vs_{}_{}_TRAIN".format(figure_path, x_var, y_var2, model_name)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)
            
            fig = sns.catplot(kind="point", x=x_var, y=y_var,  data=self.dataset.loc[test_sel,:], hue=hue, linestyles = linestyles, markers=filled_markers, markersize=10, legend=False, **kws_online_2)
            plt.xlabel(x_var, fontsize=label_fontsize)
            plt.legend(fontsize=label_fontsize - 3, frameon=True, framealpha=0.5, title=hue, ncol=4, bbox_to_anchor=(1.,1.4))
            if '%' in y_var: 
                plt.ylabel("Error (%)", fontsize=label_fontsize)
                y_var2 = y_var.replace("(%)", "perc")
            else:
                plt.ylabel("Error" + units[self.target], fontsize=label_fontsize)
            plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
            X, Y = self.dataset.loc[test_sel, x_var].values, self.dataset.loc[test_sel, y_var].values
            util.PlotGridSpacing(X, Y, x_gridno=8, y_gridno=6)
            filename_plot = "{}/{}_vs_{}_{}_TEST".format(figure_path, x_var, y_var2, model_name)
            util.SaveFigure(filename_plot, fig, isshow=isfigshow)

class CostModeler:
    """
    Cost modeling module which implements feature dataset preparation, feature analysis, and predictive modeling sub-modules.
    It prepares feature dataset from \textit{CostDataset} for modeling and analysis purposes. 
    To make the cost prediction models robust for in- and out-of-sample predictions, the targets were converted into per-unit-house values by dividing them with the number of houses. 
    """
    def __init__(self, data_path, system_name):

        self.data_path = data_path
        self.result_path = "{}/{}".format(self.data_path, setting_dict["modeling"]["results_dir"]) 
        util.CreateDir(self.result_path)

        self.figure_path = "{}/{}".format(self.result_path, setting_dict["modeling"]["figure_dir"])
        util.CreateDir(self.figure_path)

        self.models_path = "{}/{}".format(self.result_path, setting_dict["modeling"]["models_dir"])
        util.CreateDir(self.models_path)
        
        self.model_dataset_dir = "{}/{}".format(self.result_path, setting_dict["modeling"]["datasets_dir"])
        util.CreateDir(self.model_dataset_dir)

        self.result_dirs = next(os.walk(self.data_path))[1]
        
        self.filename_mldataset = "{}/{}".format(self.model_dataset_dir, setting_dict["modeling"]["dataset_file"])
        
        self.exprmnt_setting_dict = None
        self.aggregator = np.median # np.mean

        self.features = ['Number of Houses', 'Number of Processes']
         
        self.targets = ['ProcTime_FromSystem', 'ProcTime_FromMeter', 'ProcEnergy_FromMeter']

        self.sel_cols_online = ['num_processes', 'dw_min', 'num_houses', 'num_houses_chunk',
                                'num_houses_online', 'onlineId',    
                                'time_mpInit_sec',
                                'time_disagg_online_sec',
                                 'time_mpClose_sec', 
                                'time_total_online_sec', 'time_mpInit_total_sec',
                                'time_mpClose_total_sec', 'time_disagg_total_sec', 'exprmntId',
                                'time_peronline_total_sec', 'time_mp_total_sec', 'meter_Nh_chunkId',
                                'meter_time_peronline_total_sec', 
                                'meter_energy_nilm_wh', 'meter_energy_bg_wh','meter_energy_wh', 'meter_energy_bg_dw_wh',
                                'meter_power_bg_median_w', 'meter_power_nilm_median_w', 'meter_power_median_w', 'meter_power_utilization_median_perc',
                                'house_energy_min_kwh','house_energy_mean_kwh', 'house_energy_median_kwh', 'house_energy_max_kwh','house_energy_total_kwh'
                                ]
        self.plot_sel_cols_online = [ 
                                    'time_mp_total_sec',
                                    # 'time_mpInit_sec',
                                    'time_disagg_online_sec', 
                                    # 'time_mpClose_sec', 
                                    'time_total_online_sec', 
                                    'time_total_online_sec', 'time_mpInit_total_sec',
                                    'time_mpClose_total_sec', 'time_disagg_total_sec',
                                    'time_peronline_total_sec', 'time_mp_total_sec',
                                    'meter_time_peronline_total_sec', 'meter_energy_nilm_wh', 'meter_energy_bg_wh',
                                    'meter_energy_wh', 'meter_energy_bg_dw_wh', 'meter_power_bg_median_w', 'meter_power_nilm_median_w',
                                    'meter_power_median_w', 'meter_power_utilization_median_perc', 
                                    'house_energy_min_kwh','house_energy_mean_kwh', 'house_energy_median_kwh', 'house_energy_max_kwh','house_energy_total_kwh'
                                    ]

    def LoadWorkingDataset(self):
        
        dfs = []
        dfs_median = []
        for exprmntdir in util.tqdm(self.result_dirs):
            
            if not exprmntdir.startswith('costlog_'):
                continue

            filename_costdataset = "{}/{}/{}".format(self.data_path, exprmntdir, setting_dict["cost_dataset"]["raw_file"])
            filename_costdataset_agg = "{}/{}/{}".format(self.data_path , exprmntdir, setting_dict["cost_dataset"]["agg_file"]) 
 
            df_cost = pd.read_csv(filename_costdataset)
            if not df_cost.empty:
                df_cost['exprmntdir'] = exprmntdir
                dfs.append(df_cost)
            
            df_cost_agg = pd.read_csv(filename_costdataset_agg)
            if not df_cost_agg.empty:
                df_cost_agg['exprmntdir'] = exprmntdir
                dfs_median.append(df_cost_agg)

        df_cost = pd.concat(dfs, axis='rows', ignore_index=True)
        df_cost_agg = pd.concat(dfs_median, axis='rows', ignore_index=True)
        
        for col in self.plot_sel_cols_online:
            df_cost[col] = df_cost[col].astype('float')
            df_cost_agg[col] = df_cost_agg[col].astype('float')
        
        self.df_cost = df_cost
        self.df_cost_agg = df_cost_agg
        
        return df_cost, df_cost_agg
    
    def FeatureDataPrepartation(self):
        
        self.dataset = self.df_cost.copy()
        
        self.dataset = self.dataset.loc[
            ~(self.dataset["num_houses"].isin(setting_dict["modeling"]["exclude_data"]["num_houses"]) | 
                self.dataset["num_processes"].isin(setting_dict["modeling"]["exclude_data"]["num_processes"])), :]
        print("costdataset size: {}".format(self.dataset.shape))
        
        sel_vars = ['dw_min', 'num_houses', 'num_processes', 'time_peronline_total_sec', 'time_disagg_online_sec', 'meter_time_peronline_total_sec','meter_energy_nilm_wh',
             'meter_energy_bg_dw_wh', 'house_energy_min_kwh','house_energy_mean_kwh', 'house_energy_median_kwh', 'house_energy_max_kwh','house_energy_total_kwh']
        
        self.dataset = self.dataset[sel_vars].astype('float')
        self.dataset[['num_houses', 'num_processes']] = self.dataset[['num_houses', 'num_processes']].astype('int')
        # print(self.dataset.shape)
        
        self.dataset.rename(columns = { 
                                        'dw_min':'dw_minutes', 
                                        'num_houses': 'Number of Houses', 
                                        'num_processes':'Number of Processes',
                                        'time_peronline_total_sec': 'ProcTime_FromSystem',
                                        'meter_time_peronline_total_sec': 'ProcTime_FromMeter', 
                                        'meter_energy_nilm_wh': 'ProcEnergy_FromMeter'}, inplace=True)

        # self.features = ['Number of Houses', 'Number of Processes']
         
        # self.targets = ['ProcTime_FromSystem', 'ProcTime_FromMeter', 'ProcEnergy_FromMeter']
        # shuffling
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

        #Save dataset
        self.dataset.reset_index(drop=True, inplace=True)
        util.SaveDatatoCSV(self.filename_mldataset, self.dataset)
        
        self.dataset_master = self.dataset.copy()
        sel_cols_online = ['Number of Houses', 'Number of Processes']
        self.dataset = self.dataset.groupby(sel_cols_online, as_index=False).aggregate(self.aggregator)
        
        return self.dataset, self.features, self.targets    
        
    def ModelBuilding(self, features, target, isfigshow=False):
        
        self.ObjMLModel = PredictiveModel(models_path = self.models_path, dataset=self.dataset, features=features, target=target)
        
        model_name, features = self.ObjMLModel.StatBasedModel()
        self.ObjMLModel.ModelEval(model_name, features, self.figure_path, isfigshow=isfigshow)

        model_name, features = self.ObjMLModel.MLBasedModel()
        self.ObjMLModel.ModelEval(model_name, features, self.figure_path, isfigshow=isfigshow)

        return self.ObjMLModel
    
    def SaveTrainedModel(self, ObjMLModel):
        
        print('Saving model... ', ObjMLModel.filename_mlmodel)
        util.SaveJson(ObjMLModel.filename_mlmodel_eval, ObjMLModel.costmodels['evals'])

        modeldata_dict = {'models': ObjMLModel.costmodels['models'], 'features': ObjMLModel.features, 'normalizer': ObjMLModel.normalizer}
        with open(ObjMLModel.filename_mlmodel,'wb') as fhandle:
            pickle.dump(modeldata_dict, fhandle)
            fhandle.close()
    
    def LoadTrainedModel(self, target):   
        
        ObjMLModel = PredictiveModel(models_path=self.models_path, target=target)
        filename_mlmodel = ObjMLModel.filename_mlmodel
        print('Load model... ', filename_mlmodel)
        with open(filename_mlmodel,'rb') as fhandle:
            modeldata_dict = pickle.load(fhandle)
            fhandle.close()
        
        return modeldata_dict, filename_mlmodel
    
    def PredictCost(self, df_pred, target, model_name):
        
        # load train prediction models
        modeldata_dict, filename_model = self.LoadTrainedModel(target)
        ObjMlModel = modeldata_dict['models']
        features = modeldata_dict['features']
        normalizer = modeldata_dict['normalizer']

        y_pred = []
        # prediction
        try:
            if model_name == "ME":   
                def StatBasedModel_predct(x, MEModel):
                    return np.array([MEModel.loc[MEModel.index==xi].values[0,0] for xi in x])    
                X = df_pred.loc[:, features].values    
                MEModel = ObjMlModel[model_name]["model"]
                y_pred = StatBasedModel_predct(X.T[0], MEModel)
            elif model_name == "RF":
                df_pred["num_houses_inv"] = df_pred["Number of Houses"].apply(lambda x: 1/x)
                features.append("num_houses_inv")
                X = df_pred.loc[:, features].values
                mlmodel = ObjMlModel[model_name]["model"]
                y_pred = mlmodel.predict(X)
                y_pred = normalizer.inverse_transform(y_pred.reshape(-1, len(y_pred)))[0]
        except Exception as e:
            print("invalid model name: ", e.message)
        
        return y_pred
    
@util.timer
def main():
    
    util.clear()
    # selecting target cost dataset working directory
    system_name, data_path = util.GetTargetDir(main_result_path, "server name")
    print("Target directory is: " + data_path)
    
    ObjModel = CostModeler(data_path=data_path, system_name=system_name)
    df_cost, df_cost_agg = ObjModel.LoadWorkingDataset()
    dataset, features, targets = ObjModel.FeatureDataPrepartation()
    
    # building cost model for processing time
    print("{}".format("#"*80))
    target = 'ProcTime_FromMeter'
    features = ObjModel.features
    print("Model building for {}...".format(target))
    print("\nfeatures: {}".format(features))
    ObjMlModel_PT = ObjModel.ModelBuilding(features, target)
    ObjModel.SaveTrainedModel(ObjMlModel_PT)
    print("\n{}".format("#"*80))
    
    # building cost model for processing energy
    print("{}".format("#"*80))
    target = 'ProcEnergy_FromMeter'
    print("Model building for {}...".format(target))
    print("\nfeatures: {}".format(features))
    ObjMlModel_PE = ObjModel.ModelBuilding(features, target)
    ObjModel.SaveTrainedModel(ObjMlModel_PE)
    print("\n{}".format("#"*80))

if __name__ == '__main__': 

    main()

    msg = "{}\n The models and prediction results can be found in /{}.\n{}"\
        .format("*"*60, setting_dict["modeling"]["results_dir"], "*"*60)

    print(msg)