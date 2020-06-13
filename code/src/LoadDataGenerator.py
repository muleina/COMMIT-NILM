"""
Created on Fri Aug 30 17:31:26 2019

@author: Mulugeta W.Asres
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, gc
import numpy as np, pandas as pd
import datetime, time
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

# %% Target NILM algorithm integrations
from TargetAlgNILMRunner import TargetAlgRunner as AlgRunner
da = AlgRunner(setting_dict["db_info"]["conn"]).ConnDb()

#%% plot settings
plt.rcParams['figure.figsize'] = (15,  3)

# %%
class DataCleaner(object):
    """
    Provides data cleaning tools for preparing the cleaned data CSV file with cleaned daily load data db references. 
    It cleaned any missing and gaps in the timestamps and prepared daily formated data with reference house ids and start and end time stamps.
    """
    def __init__(self, datatimestamp_splitsize=1):
        
        self.datatimestamp_splitsize = datatimestamp_splitsize # in days
        data_ref = np.squeeze(setting_dict["db_info"]["data"]["values"])
        self.houseids = list(data_ref[:, 0].astype(setting_dict["db_info"]["data"]["vartypes"][0]))
        self.start_datetimes = list(data_ref[:, 1].astype(setting_dict["db_info"]["data"]["vartypes"][1]))
        self.end_datetimes = list(data_ref[:, 2].astype(setting_dict["db_info"]["data"]["vartypes"][2]))
        self.cleaneddata_db_ref_file = "{}/{}".format(data_path, setting_dict["db_info"]["cleaneddata_db_ref_file"])
        
        self.cleaneddata_db_ref = pd.DataFrame()
        self.cleaneddata = dict()
        
        if not da.status:
            raise "Db connection error!"
            
        print("Seting: retrieve data from {} unique load houseids.".format(len(self.houseids)))
        
    def _PrepareBaselineDataRef(self):
        
        self.data_db_ref = dict()
        self.start_date_online_all = dict()
        
        for i, houseid in enumerate(self.houseids):
            self.data_db_ref[houseid] = [datetime.datetime.strptime(self.start_datetimes[i], "%Y-%m-%d %H:%M:%S"), datetime.datetime.strptime(self.end_datetimes[i], "%Y-%m-%d %H:%M:%S")]
            self.start_date_online_all[houseid] = [datetime.datetime.strptime(self.start_datetimes[i], "%Y-%m-%d %H:%M:%S"), 0]
        
    def _LocateMissingData(self, data_timestamps):
        
        data_timestamps['timestamp'] = data_timestamps.index
        data_timestamps.reset_index(drop=False, inplace=True)
        data_timestamps['diff_timestamp'] = data_timestamps.timestamp.diff().fillna(0).astype('timedelta64[ns]')
        data_timestamps.head()

        data_timestamps['diff_timestamp_hours'] = data_timestamps['diff_timestamp'].copy()
        data_timestamps['diff_timestamp_hours'] = data_timestamps['diff_timestamp']/ np.timedelta64(1, 'h')

        corrupted_data = data_timestamps.loc[data_timestamps['diff_timestamp'] > np.timedelta64(1, 'h'), ['timestamp', 'diff_timestamp', 'diff_timestamp_hours']]
        if not corrupted_data.empty:
            print("corrupted_data: size={}".format(corrupted_data.shape[0]))
            print(corrupted_data.head())

        return data_timestamps
    
    def _DataTimestampSplit(self, data_timestamps):
        
        Readingtime_df = data_timestamps.set_index('timestamp', inplace=False)
        cleaneddata_splited = [g for n, g in Readingtime_df.groupby(pd.TimeGrouper("{}D".format(self.datatimestamp_splitsize)))]
        
        Nw = len(cleaneddata_splited)
        cleaneddata_index = []
        for d_idx in range(0,Nw):
            # print(d_idx)
            if not cleaneddata_splited[d_idx].empty:
                # print([cleaneddata_splited[d_idx].index[0], cleaneddata_splited[d_idx].index[-1]])
                if (cleaneddata_splited[d_idx].index[0].hour < 1) & (cleaneddata_splited[d_idx].index[-1].hour > 22) & np.all(cleaneddata_splited[d_idx].loc[:, 'diff_timestamp_hours']<1):
                    cleaneddata_index.append(d_idx) # print('***Cleaned***')
        
        return cleaneddata_splited, cleaneddata_index
    
    def _CleanDataLoad(self, houseid, cleaneddata_splited, cleaneddata_index):
        
        print('Number of cleaned days: ',   len(cleaneddata_index))
        
        startend_datetimes = []
        for d_idx in cleaneddata_index:
            startend_datetimes.append([houseid, cleaneddata_splited[d_idx].index[0], cleaneddata_splited[d_idx].index[-1], \
                cleaneddata_splited[d_idx]['Mains_Power'].min(), cleaneddata_splited[d_idx]['Mains_Power'].mean(), cleaneddata_splited[d_idx]['Mains_Power'].median(), cleaneddata_splited[d_idx]['Mains_Power'].max()])
        cleaneddata_db_ref = pd.DataFrame(startend_datetimes, columns=['houseid', 'start_datetime', 'end_datetime', 'mains_power_min', 'mains_power_avg', 'mains_power_median', 'mains_power_max'])
        
        return cleaneddata_db_ref
                        
    def GenerateCleanedDataDbRef(self):
        
        self._PrepareBaselineDataRef()
        
        data_timestamps = None
        gc.collect()
        dfs = []
        
        for houseid in self.houseids:
            print("#"*70)
            print("houseid: {}".format(houseid))
            # timestamp_limits = da.getMinMaxDates(houseid) # get start and end of load logging datetimes
            # print(timestamp_limits)
            
            data_timestamps = da.getAllDatetimes(houseid)  # get all load logging datetimes
            data_timestamps.index = data_timestamps.index.astype('datetime64[ns]')
            # data_timestamps.head()
            data_timestamps = self._LocateMissingData(data_timestamps)
            cleaneddata_splited, cleaneddata_index = self._DataTimestampSplit(data_timestamps)
            self.cleaneddata[houseid] = dict()
            self.cleaneddata[houseid]["cleaneddata_index"] = cleaneddata_index
            self.cleaneddata[houseid]["cleaneddata_splited"] = cleaneddata_splited
            
            cleaneddata_db_ref_perid = self._CleanDataLoad(houseid, cleaneddata_splited, cleaneddata_index)
            dfs.append(cleaneddata_db_ref_perid)
            
        self.cleaneddata_db_ref = pd.concat(dfs, axis='rows', ignore_index=True)
        util.SaveDatatoCSV(self.cleaneddata_db_ref_file, self.cleaneddata_db_ref)

    def DisplayInfo(self):
        
        print("houseIds: {}".format(self.houseids))
        print("start_datetimes: {}".format(self.start_datetimes))
        print("end_datetimes: {}".format(self.end_datetimes))
        print("cleaneddata_db_ref_filename: {}".format(self.cleaneddata_db_ref_file))
        return self.cleaneddata_db_ref.head(100)
    
    def PlotSplitedData(self, houseids=[], max_figs=10):
        
        if len(houseids) < 1:
            houseids = self.houseids
          
        for houseid in houseids:
            print("Plot timestamp splits for house id {} ...".format(houseid))
            cleaneddata_index = self.cleaneddata[houseid]["cleaneddata_index"]
            cleaneddata_splited = self.cleaneddata[houseid]["cleaneddata_splited"]
            
            split_no =  0
            for d_idx in cleaneddata_index:
                if split_no >= max_figs:
                    break
                split_no =  split_no + 1
                data_splitted = cleaneddata_splited[d_idx]
                plt.plot(data_splitted.index, data_splitted['Mains_Power'], color='red')
                plt.title("timesplit_{}".format(split_no))
                plt.show()

    def LoadCleanedDataDbRef(self):
        
        try:
            self.cleaneddata_db_ref = pd.read_csv(self.cleaneddata_db_ref_file) 
        except Exception as ex:
            print("{}. It looks like the cleaned data ref file is not generated; run GenerateCleanedDataDbRef first.".format(ex)) 
            
        return self.cleaneddata_db_ref
      
class LargeScaleDataGenerator(DataCleaner):
    """
    Generates the large scale simulated house energy data-set to provide computation evaluation at a large scale deployment of the target NILM algorithm. 
    In our example experiment, it generates from SQL database containing public NILM data-sets with real house energy profiles, i.e., REDD and UK-DALE.
    Generally, it applies data cleaning, data chunking, shuffling, and duplication techniques on the public datasets to synthesize a large number of houses' power demands. 
    Instead of working with actual load data from the database (DB), it manipulates the data references such as the list of house ids (id in DB), start and end date-times (a reference to recording DateTime in DB).
    This approach speeds-up the data generation as well as incorporates database interfacing into the NILM cost monitoring representing real-world implementation.
    """
    def __init__(self, timewindow_slicesize=60, timewindow_unit=0, isshuffle=True):

        super(LargeScaleDataGenerator, self).__init__()

        self.isshuffle = isshuffle
        self.timewindow_slicesize = timewindow_slicesize
        self.timewindow_unit = timewindow_unit
        self.num_dataduplication = None
        self.timewindow_unit_isindays = 0
        
        if self.timewindow_unit==1:
            self.timewindow_unit_isindays = 1
        
    def _GetCleanedDataDateTimes(self):
        
	    self.data_db_ref_source = self.LoadCleanedDataDbRef()
        
    def _PrepareBaselineLoadDate(self):
        
        house_ids_all, start_datetimes, end_datetimes = \
            self.data_db_ref_source['houseid'].astype(str).values, self.data_db_ref_source['start_datetime'].values, self.data_db_ref_source['end_datetime'].values
        
        dataset_org = dict()
        start_datetimes_online_all = dict()
        end_datetimes_online_all = dict()
        cnts = dict()
        
        Nh = len(house_ids_all)
        for i in range(0, Nh):
            cnts[house_ids_all[i]]   =  0
        
        house_ids_all_new = []
        for i in range(0, Nh):
            house_id = house_ids_all[i]
            house_id_new = house_id + "__" + str(cnts[house_id])
            
            while house_id_new in dataset_org.keys():
                cnts[house_id]  = cnts[house_id]  + 1 
                house_id_new = house_id + "__" + str(cnts[house_id]) 
            
            house_ids_all_new.append(house_id_new)     
            dataset_org[house_id_new] = [datetime.datetime.strptime(start_datetimes[i], "%Y-%m-%d %H:%M:%S"), datetime.datetime.strptime(end_datetimes[i], "%Y-%m-%d %H:%M:%S")]
            start_datetimes_online_all[house_id_new] = [dataset_org[house_id_new][0], 0]
            end_datetimes_online_all[house_id_new] = [dataset_org[house_id_new][1], 0]
        
        # print(len(dataset_org.values()))
        # print(len(start_datetimes_online_all.values()))
        # print(len(end_datetimes_online_all.values()))
        # for k, v in dataset_org.items():
        #     print(k, v)
        self.data_db_ref, self.house_ids_all, self.start_datetimes_online_all, self.end_datetimes_online_all = \
            dataset_org, house_ids_all_new, start_datetimes_online_all, end_datetimes_online_all

    def _GenerateTimeSlicedLoadData(self):
        
        dataset_org, house_ids_all, start_datetimes_online_all, end_datetimes_online_all = \
             self.data_db_ref.copy(), self.house_ids_all, self.start_datetimes_online_all, self.end_datetimes_online_all

        timewindow_slicesize =  self.timewindow_slicesize
        timewindow_unit_isindays = self.timewindow_unit_isindays
             
        cnt = 1
        data_db_ref = dict()
        start_datetimes_online = start_datetimes_online_all

        house_ids = house_ids_all[:]
        # print(house_ids)
        
        while len(house_ids) > 0:        
            # print(house_ids)
            
            for i in range(0, len(house_ids)): 
                
                if timewindow_unit_isindays==1:
                    data_db_ref[house_ids[i]+"__" + str(cnt)]  = [start_datetimes_online[house_ids[i]][0], 
                                np.min([end_datetimes_online_all[house_ids[i]][0], start_datetimes_online[house_ids[i]][0] + datetime.timedelta(days = timewindow_slicesize)])]
                else:                   
                    data_db_ref[house_ids[i]+"__" + str(cnt)]  = [start_datetimes_online[house_ids[i]][0], 
                                np.min([end_datetimes_online_all[house_ids[i]][0], start_datetimes_online[house_ids[i]][0] + datetime.timedelta(minutes = timewindow_slicesize)])]
                        
            # start_datetimes_online_prev = start_datetimes_online
            start_datetimes_online={house_id: util.UpdateTimeWindowStart(v, timewindow_unit_isindays, timewindow_slicesize, house_id in house_ids) 
                                for house_id, v in start_datetimes_online.items()}

            house_ids = [house_id for house_id, v in start_datetimes_online.items() if (house_id in house_ids) and (v[0]  < dataset_org[house_id][1])]
            cnt = cnt + 1
            
        house_ids_all = list(data_db_ref.keys())
        print('Number of baseline houses generated using the disaggrgation time slot: ', len(house_ids_all))
        
        # print(' ****************************** GENERATED HOUSES ******************************')     
        start_datetimes_online_all = dict()
        end_datetimes_online_all = dict()
        
        for k, v in data_db_ref.items():
            # print(k, v)
            start_datetimes_online_all[k] = [v[0], 0]
            end_datetimes_online_all[k] = [v[1], 0]
        
        self.data_db_ref_ts, self.house_ids_all_ts, self.start_datetimes_online_all_ts, self.end_datetimes_online_all_ts = \
            data_db_ref, house_ids_all, start_datetimes_online_all, end_datetimes_online_all

    def _GenerateDuplicatedLoadData(self):         
        
        house_ids_all, start_datetimes_online_all, end_datetimes_online_all = \
             self.house_ids_all_ts, self.start_datetimes_online_all_ts, self.end_datetimes_online_all_ts
        
        num_dataduplication = self.num_dataduplication
        
        # house_ids_all_org = house_ids_all*num_dataduplication
        house_ids_all_org = []
        
        for i in range(num_dataduplication):
            np.random.shuffle(house_ids_all)
            house_ids_all_org.extend(house_ids_all)

        house_ids_all = np.arange(1, len(house_ids_all)*num_dataduplication + 1)
        
        start_datetimes = [start_datetimes_online_all[k][0] for k in house_ids_all_org]
        end_datetimes = [end_datetimes_online_all[k][0] for k in house_ids_all_org]
        data_db_ref = dict()
        start_datetimes_online_all = dict()
        
        for i in range(0, len(house_ids_all)):
            data_db_ref[house_ids_all[i]] = [start_datetimes[i], end_datetimes[i]]
            start_datetimes_online_all[house_ids_all[i]] = [start_datetimes[i], 0]
        gc.collect()
        print('Number of generated duplicated houses for the experiment: ', len(house_ids_all_org))
        
        self.data_db_ref_ts_dp, self.house_ids_all_org_ts_dp, self.house_ids_all_ts_dp, self.start_datetimes_online_all_ts_dp, self.start_datetimes_ts_dp, self.end_datetimes_ts_dp = \
        data_db_ref, house_ids_all_org, house_ids_all, start_datetimes_online_all, start_datetimes, end_datetimes
               
    def LargeScaleLoadData(self, target_datasize):
        """Generating experimental data for the given maximum number of houses and disaggregation time-window."""
        
        print('Generating experimental data for the given maximum number of houses...')

        # Get the baseline cleaned dataload db reference
        self._GetCleanedDataDateTimes()
        
        # prepare the baseline data load db ref variables and rename dataload id into the format of id_count_num 
        self._PrepareBaselineLoadDate()
        
        # Generating data load using time slices using the given NILM disaggregation time window
        self._GenerateTimeSlicedLoadData()
        
        self.target_datasize = target_datasize

        # estimate the number of required duplications to mmet the target load scale
        self.num_dataduplication = int(np.ceil(float(self.target_datasize)/len(self.house_ids_all_ts)))

        # shuffling to introduce randomness
        if self.isshuffle:
            np.random.shuffle(self.house_ids_all_ts)  

        if self.num_dataduplication > 0:
            # Duplicating load data
            self._GenerateDuplicatedLoadData()
        else:
            self.data_db_ref_ts_dp, self.house_ids_all_org_ts_dp, self.house_ids_all_ts_dp, self.start_datetimes_online_all_ts_dp, self.start_datetimes_ts_dp, self.end_datetimes_ts_dp = \
            self.data_db_ref_ts, self.house_ids_all_org_ts, self.house_ids_all_ts, self.start_datetimes_online_all_ts, self.start_datetimes_ts, self.end_datetimes_ts
                     
        
        self.generated_datasize = len(self.house_ids_all_ts_dp)
        
        # return self.data_db_ref_ts_dp, self.house_ids_all_org_ts_dp, self.house_ids_all_ts_dp, self.start_datetimes_online_all_ts_dp, self.start_datetimes_ts_dp, self.end_datetimes_ts_dp
    
    def DisplayGeneratedData(self, level = 'dp'):
        
        if level == 'base':
            house_ids_all =  self.house_ids_all
            house_ids_all_org = self.house_ids_all
            data_db_ref = self.data_db_ref
        elif level == "ts":
            house_ids_all =  self.house_ids_all_ts
            house_ids_all_org = self.house_ids_all_ts
            data_db_ref = self.data_db_ref_ts
        elif level == "dp":
            house_ids_all =  self.house_ids_all_ts_dp
            house_ids_all_org = self.house_ids_all_org_ts_dp
            data_db_ref = self.data_db_ref_ts_dp
        else:
            print('Please select the level that \"base\": baseline data, \"ts\": timesliced and \"dp\": duplicated data!')
            return
        
        for id_h in house_ids_all:
            print([id_h, house_ids_all_org[id_h-1], data_db_ref[id_h]])

@util.timer
def main():

    util.clear()
    
    print('house load data cleaning...')

    # initialization of data cleaner
    ObjDataCleaner = DataCleaner(datatimestamp_splitsize = 1) 

    # display house load data setting info
    ObjDataCleaner.DisplayInfo()

    # clean and prepared the baseline daily cleaned load data ref dataset and save it for later uses in large-scale laod generation
    ObjDataCleaner.GenerateCleanedDataDbRef()

    # load the generated cleaned load dataset
    cleaneddata_db_ref = ObjDataCleaner.LoadCleanedDataDbRef()
    
    # display house load dataset info
    ObjDataCleaner.DisplayInfo()

if __name__ == "__main__":

    main()

    msg = "{}\n The resulting cleaned laod data db references can be found in {}.\n{}"\
        .format("*"*60, setting_dict["db_info"]["cleaneddata_db_ref_file"], "*"*60)

    print(msg)
    