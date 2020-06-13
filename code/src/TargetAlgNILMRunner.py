"""
Created on Fri Apr 3 17:33:21 2020

@author: Mulugeta W.Asres

Customize this file while keeping the fuction names and iterfaces as there are to integarate the CCMT with your own NILM Algorithm

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc
import numpy as np, pandas as pd
import time

# %%
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_path)
import utilities as util

base_path = os.path.dirname(os.getcwd())
code_path, data_path, result_path = util.GetWorkingDirs(base_path) 
sys.path.append(base_path)
sys.path.append(code_path)

setting_dict  = util.LoadJson("{}/setting.json".format(src_path))

#%% libraries specific to your NILM algorithm
from NILMAlgorithm import *

# %%
class DataAccess:
    """
    Connect to db for load data access.
    Change this class functions content while keep the interfaces for your own db connection function.
    """
    def __init__(self, db_info):
        """
        # initialize db connection load data access.
        """
        self.db_server_ip = db_info["server_ip"]
        self.db_user = db_info["user"]
        self.db_pass = db_info["pass"]
        self.db_name = db_info["db"]
        
        self.status = False
        '''
        # connect to db here for load data access. e.g. using pymssql
        try:
            self.conn = pymssql.connect(erver=self.db_server_ip, user=self.db_user, pwd=self.db_pass, database=self.db_name)
            self.status = True
        except Exception as ex:
            raise "Db connection error. {ex}!".format(ex)
        '''
		
        print("\nTarget NILM algorithm is not uploaded or database connection is not defined! Please update TargetAlgNILMRunner.py first and follow the instruction!\n")
    
    def getAllDatetimes(self, houseid):
        """
        for accessing all load recording datatimes during LoadDataGeneration.
        """
        print("\nTarget NILM algorithm is not uploaded or database connection is not defined! Please update TargetAlgNILMRunner.py first and follow the instruction!\n")
        return record_dates

    def getLoadData(self, house_id, start_datetime, dw):
        """
        for accessing all load recording datatimes during LoadDataGeneration.
        """
        print("\nTarget NILM algorithm is not uploaded or database connection is not defined! Please update TargetAlgNILMRunner.py first and follow the instruction!\n")
        return df_load

    def cleanResultLogTable(self):
        """
        for accessing all load recording datatimes during LoadDataGeneration.
        """
	
class TargetAlgRunner:
    """
    Provides an interface to the target NILM algorithm under monitoring. 
    It embarks with database connection parameters defined in the setting.JSON,  ConnDb(), to initialize connection to the database, and finally, uses \textit{Runner()} for calling the target functions main function, i.e., load disaggregation. 
    To effectively employ the CCMT, users need to customize the implementation of each function block of this module according to their algorithm, while keep the function names consistent. 
    """
    def __init__(self, db_info):
        """
        database access information
        """
        self.db_info = db_info

    def ConnDb(self):
        """ 
        Connect to db for load data access.
        Change this section for your own db connection function.

        da is a data access class object with interfaces of getAllDatetimes(houseid), GetLoadData() and cleanResultLogTable().
        da.getAllDatetimes(houseid): for accessing all load recording datatimes during LoadDataGeneration.
        da.getLoadData(house_id, start_datetime, dw): for retrieving the time-window load data from db.
        da.cleanResultLogTable(): (optional) to clean previous results of the NILM alg from db to prevent unbounded growing of data in db.
        """

        da = DataAccess(self.db_info)

        return da

    def Runner(self, dw, house_id, house_id_org, start_datetime):
        """
        Calls the algorithm under monitoring. 
        Change this section keeping the template for your own NILM algorithm
        """
        house_energy = 0
        pt_dict = {}

        # db connection time
        t_start = time.time() 
        # connecting to server db
        da = self.ConnDb()
        pt_dict["time_preProcessing_sec"] = time.time() - t_start

        # setting loading time
        t_start = time.time() 
        # load settings
        '''  
        # load house specific settings if there are any
        setting
        '''
        pt_dict["time_loadSetting_sec"] = time.time() - t_start

        # model loading time
        t_start = time.time() 
        # load appliance models
        '''  
        # load house specific appliance models if there are any
        appmodels
        '''
        pt_dict["time_loadModel_sec"] = time.time() - t_start

        # load disagg time for the given houseid and timewindow 
        t_start = time.time()

        # get the time-window load data from db
        df_load = da.getLoadData(house_id, start_datetime, dw)
        '''
        # calculate energy in kwh from df_load
        house_energy = calcEnergy(df_load['t'], df_load['P'])

        # call load disaggegation function (NILM algorithm)
        nilm_result, _ = NILMAlg(df_load, setting, appmodels)
        ''' 
        pt_dict["time_disagg_sec"] = time.time() - t_start

        # total energy of per disaggWindow
        pt_dict["energy_kwh"] = house_energy

        return pt_dict

    def CleanOldResultLogs(self):
        """
        (Optional) to clean previous disaggregation results from db to prevent unbounded growing of disagg result in db
        """
        print('cleaning previous disaggregation results from db...')
        da = self.ConnDb()
        da.cleanResultLogTable()
