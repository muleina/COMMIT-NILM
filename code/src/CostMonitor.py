"""
Created on Fri Aug 30 17:33:21 2019

@author: Mulugeta W.Asres

"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, psutil, gc, json, random
import numpy as np, pandas as pd 
import datetime, time, calendar
import multiprocessing as mp
from collections import OrderedDict

# %%
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_path)
import utilities as util

base_path = os.path.dirname(os.getcwd())
setting_dict  = util.LoadJson("{}/setting.json".format(src_path))

# %%
try:
    system_name = os.environ['COMPUTERNAME']
except:
    system_name_ids = [int(name.split('_')[1]) for name in next(os.walk(base_path + "results"))[1] if name.startswith("unknownserver_")]
    if len(system_name_ids) > 0:
        system_name_id = np.max(system_name_ids) + 1
    else:
        system_name_id = 1

    system_name = "unknownserver_{}".format(system_name_id)
    
print('SYSTEM:', system_name)

code_path, data_path, result_path = util.GetWorkingDirs(base_path, system_name) 

sys.path.append(base_path)
sys.path.append(code_path)

util.clear()
affix = '_'

# %% Target NILM algorithm integrations
from TargetAlgNILMRunner import TargetAlgRunner as AlgRunner
nilmAlg = AlgRunner(setting_dict["db_info"]["conn"])

# %% Load experimrnt setting
assert(setting_dict["num_repeated_expt"] >= 1, "num_repeated_expt can not be less than one.")	
assert(setting_dict["max_onlineunique_loads"] >= 1, "max_onlineunique_loads can not be less than one.")	
assert(setting_dict["num_processors"]["start"] >= 1, "min_num_processors can not be less than one.")	
assert(setting_dict["synch_duration"] >= 3, "synch_duration can not be less than 3 seconds for robust synchronization patterns.")

num_repeated_expt = setting_dict["num_repeated_expt"]
Nh_chunks = list(np.arange(setting_dict["num_houses"]["start"],  setting_dict["num_houses"]["end"]+1,  setting_dict["num_houses"]["step"]))
Nh_chunks = list(np.sort(Nh_chunks))
Nh_chunk_max = np.max(Nh_chunks)

Nps = np.arange(setting_dict["num_processors"]["start"],  setting_dict["num_processors"]["end"]+1,  setting_dict["num_processors"]["step"])
if Nps[-1] > mp.cpu_count():
    warnings.warn("Running parallel processing greater than Your system's processors ({}) is not recommended.".format(mp.cpu_count()))

isdays = setting_dict["disaggregation_window_isperday"] # 0 for disaggtimewindow in minutes, 1 disaggtimewindow in days
disaggtimewindow = setting_dict["disaggregation_window"] # 30, 45, 60 min data segment selection

#  for cluster-based approach
synch_len = setting_dict["synch_duration"]
init_d = synch_len
exprmnt_d = synch_len
Nhchunk_d = synch_len
Np_d = synch_len
onlinedw_d = synch_len
isoneonline = 1 # 0 for multiple onlines based on disaggtimewindow, 1 put all dwspans in single online: creates more houses

#%% Generate large scale houses loads for the given maximum number of houses and NILM algorithm's disaggregation window
from LoadDataGenerator import LargeScaleDataGenerator
dataGenerator = LargeScaleDataGenerator(timewindow_slicesize=disaggtimewindow, timewindow_unit=isdays, isshuffle=True)
dataGenerator.LargeScaleLoadData(target_datasize=Nh_chunk_max)

num_houseload_duplication = dataGenerator.num_dataduplication
dataset = dataGenerator.data_db_ref_ts_dp
house_ids_all_org = dataGenerator.house_ids_all_org_ts_dp
house_ids_all = dataGenerator.house_ids_all_ts_dp
start_datetimes_online_all = dataGenerator.start_datetimes_online_all_ts_dp

print("Number of load data duplications: ", num_houseload_duplication)

'''
for id_h in house_ids_all:
    print([id_h, house_ids_all_org[id_h-1], dataset[id_h]])
time.sleep(1)
'''
# %%
def CreateWorkingDirs(result_path, name_ref="costlog_"):
    """create new log direcories specific for current experiment."""
     
    print("preparing log directories...")
    util.CreateDir(result_path)
    prev_costlog_ids = [int(name.split('_')[1]) for name in next(os.walk(result_path))[1] if name.startswith(name_ref)]
    if len(prev_costlog_ids) > 0:
        costlog_id = np.max(prev_costlog_ids) + 1
    else:
        costlog_id = 1

    costlog_dir =  "{}_{}".format(name_ref, costlog_id)
    costlog_path = "{}/{}".format(result_path, costlog_dir)
    meter_log_path = "{}/{}".format(costlog_path, setting_dict["meter_log"]["log_dir"])
    system_log_path = "{}/{}".format(costlog_path, setting_dict["system_log"]["log_dir"])

    util.CreateDir(costlog_path)
    util.CreateDir(meter_log_path)
    util.CreateDir(system_log_path)

    print("done!")
    return costlog_path, system_log_path, meter_log_path
    
def SyncPatternInsertion(duration_sec=3):
    """generates and inserts syncronization pattern signal into the system-level power reading."""
     
    def high(duration_sec):
        timeout = time.time() + duration_sec
        while True:
            if time.time() > timeout:
                break
    
    def low(duration_sec):
        time.sleep(duration_sec)

    low(2*duration_sec) # sleep
    high(duration_sec) # awake
    low(2*duration_sec) # sleep

def getProcessLevelInfo(process):
    """get process-level resource instant consumption."""
     
    resource_dict = {}
    with process.oneshot(): 
        p_cputime = process.cpu_times()
        p_mem = process.memory_full_info()
        p_diskio = process.io_counters()
        
        resource_dict["p_cpu_t.user_sec"] = p_cputime.user 
        resource_dict["p_cpu_t.sys_sec"] = p_cputime.system 
        p_cputime_user = p_cputime.user
        p_cputime_system = p_cputime.system
        resource_dict["p_cpu_t.us_sec"] = p_cputime_user + p_cputime_system 
        resource_dict["p_cpu_perc"] = process.cpu_percent(interval=None)   # return cached value
       
        resource_dict["p_mem.rss_MB"] = p_mem.rss/float(2**20) 
        resource_dict["p_mem.vms_MB"] = p_mem.vms/float(2**20) 
        resource_dict["p_mem.uss_MB"] = p_mem.uss/float(2**20) 
        resource_dict["p_mem_per"] = process.memory_percent(memtype="rss") #Compare process memory to total physical system memory and calculate process memory utilization as a percentage
       
        resource_dict["p_diskio.rd_cnt"] = p_diskio.read_count  #the number of read operations performed (cumulative)
        resource_dict["p_diskio.wr_cnt"] = p_diskio.write_count  #the number of write operations performed (cumulative)
        resource_dict["p_diskio.rd_MB"] = p_diskio.read_bytes/float(2**20)  #the number of bytes read (cumulative)
        resource_dict["p_diskio.wr_MB"] = p_diskio.write_bytes/float(2**20)  #the number of bytes written (cumulative)
        resource_dict["p_diskio.other_cnt"] = p_diskio.other_count  # only Windows, the number of I/O operations performed other than read and write operations.
        resource_dict["p_diskio.other_MB"] = p_diskio.other_bytes/float(2**20)  #only Windows, the number of bytes transferred during operations other than read and write operations.
        resource_dict["p_numCPUAffinity"] = len(process.cpu_affinity()) 
        resource_dict["p_numChildP"] = len(process.children())  # get childen subprocess list
        resource_dict["p_numChildP_deep"] = len(process.children(recursive=True))  # including all grand and great grand childrens 
        resource_dict["p_num_threads"] = process.num_threads()  # not cumulative
        resource_dict["p_num_handles"] = process.num_handles()  # not cumulative
    
    return resource_dict

def getSystemLevelInfo():
    """get system-level resource instant consumption."""
    
    resource_dict = {}
    cputime = psutil.cpu_times()
    memvir = psutil.virtual_memory()
    memswap = psutil.swap_memory()
    diskio = psutil.disk_io_counters(perdisk=False)

    cputime_user = cputime.user 
    cputime_sys = cputime.system 
    resource_dict["sys_cpu_t.user_sec"] = cputime_user
    resource_dict["sys_cpu_t.sys_sec"] = cputime_sys
    resource_dict["sys_cpu_t.us_sec"] = cputime_user + cputime_sys
    resource_dict["sys_cpu_t.idle_sec"] = cputime.idle
    resource_dict["sys_cpu_t.interrupt_sec"] = cputime.interrupt
    resource_dict["sys_cpu_t.dpc_sec"] = cputime.dpc
    resource_dict["sys_cpu_perc"] = psutil.cpu_percent(interval=None)
    
    # print(memvir)
    resource_dict["sys_memvir.used_MB"] = memvir.used/float(2**20)
    resource_dict["sys_memvir.perc"] = memvir.percent
   
    resource_dict["sys_memswap.used_MB"] = memswap.used/float(2**20)
    resource_dict["sys_memswap.perc"] = memswap.percent
   
    # print(diskio)
    resource_dict["sys_diskio.rd_cnt"] = diskio.read_count
    resource_dict["sys_diskio.wr_cnt"] = diskio.write_count
    resource_dict["sys_diskio.rd_MB"] = diskio.read_bytes/float(2**20)
    resource_dict["sys_diskio.wr_MB"] = diskio.write_bytes/float(2**20)
    resource_dict["sys_diskio.rd_t_sec"] = diskio.read_time/float(1000)
    resource_dict["sys_diskio.wr_t_sec"] = diskio.write_time/float(1000)
    
    return resource_dict

def GetNextOnlineData(house_ids, start_datetimes_online, dataset, dw):
    
    # additional code for creating the dw slot groups
    start_datetimes_online={house_id: util.UpdateTimeWindowStart(v, isdays, dw, house_id in house_ids) 
                        for house_id, v in start_datetimes_online.iteritems() }
    house_ids = [house_id for house_id, v in start_datetimes_online.iteritems() if (house_id in house_ids) and (v[0]  < dataset[house_id][1])]
    
    return house_ids, start_datetimes_online  

def AlgorithmMonitor(Np, dw, Nh, Nh_chunk, Nh_online, onlineId, house_id, house_id_org, start_datetime_1):
    """call target NILM algorithm disaggregation function or Runner() to for cost monitoring."""
    
    process = psutil.Process(os.getpid())
    house_id_org_generated = house_id_org
    house_id_names = house_id_org_generated.split('__', 1)
    house_id_org = house_id_names[0]
    print("House_id: ", house_id)

    pt_dict = {
        "processid":process.pid, 
        "numProcesses":Np, "dw_min":dw, "num_houses":Nh, "num_houses_chunk":Nh_chunk, "num_houses_online":Nh_online, 
        "onlineId":onlineId, 
        "houseId_local":house_id, "houseId_org_gen":house_id_org_generated, "houseId_org":house_id_org, 
        "dw_start_datetime":start_datetime_1
    }

    # call NILM algorithm to disaggregate the power consumption of house_id_org at start_datetime_1
    pt_dict_nilm = nilmAlg.Runner(dw, house_id, house_id_org, start_datetime_1)

    pt_dict.update(pt_dict_nilm)
            
    return pt_dict

def SerialProcessing(Np, dw, Nh_chunk, house_ids):   
    """running the NILM algorithm in single process"""
    
    house_ids = house_ids[0:Nh_chunk]
    Nh = len(house_ids)
    dfs_process =[]
    pt_sys_all = []
    onlineId = 0
    pt_disagg_total = 0
    pt_mp_init_total = 0
    pt_mp_close_total = 0
    start_datetimes_online = start_datetimes_online_all

    t_start = time.time()

    while len(house_ids) > 0:  
        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(onlinedw_d)

        t_start_online = time.time()
        onlineId = onlineId + 1
        Nh_online = len(house_ids)
        pt_sys_dict = {
                        "numProcesses":Np, "dw_min":dw, 
                        "num_houses":Nh, "num_houses_chunk":Nh_chunk, "num_houses_online":Nh_online, 
                        "onlineId":onlineId
                    }

        t_start = time.time()
        pool = mp.Pool(processes=Np)
        pt_mp_init = time.time() - t_start
        
        # pt_sys_init = getSystemLevelInfo()

        t_start = time.time()
        output = [AlgorithmMonitor(Np, dw, Nh, Nh_chunk, Nh_online, onlineId, house_id, house_ids_all_org[house_id-1], start_datetimes_online[house_id][0]) 
                    for house_id in house_ids]

        dfs_process.append(pd.DataFrame(output))

        pt_disagg_online = time.time() - t_start
       
        pt_mp_close = 0

        pt_mp_init_total = pt_mp_init_total + pt_mp_init
        pt_disagg_total = pt_disagg_total + pt_disagg_online
        pt_mp_close_total = pt_mp_close_total + pt_mp_close

        # pt_sys_final = getSystemLevelInfo()
        # pt_sys_d = [pt_sys_final[s] - pt_sys_init[s] for s in range(len(pt_sys_init))]

        # additional code for creating the dw slot groups, simultes db update for the next reading
        house_ids, start_datetimes_online = GetNextOnlineData(house_ids, start_datetimes_online, dataset, dw)
        
        # pt_sys.append(pt_online)

        pt_sys_dict["time_mpInit_sec"] = pt_mp_init
        pt_sys_dict["time_disagg_online_sec"] = pt_disagg_online
        pt_sys_dict["time_mpClose_sec"] = pt_mp_close
        # pt_sys.extend(pt_sys_d)
        # pt_sys.extend(pt_sys_final)

        pt_online = time.time() - t_start_online
        pt_sys_dict["time_total_online_sec"] = pt_online

        pt_sys_all.append(pt_sys_dict)     

        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(onlinedw_d)

    #  Combining all process based results
    p_info_df_all = pd.concat(dfs_process, axis='rows', ignore_index=True)
    p_info_df_all['time_disagg_total_sec'] = pt_disagg_total

    # Combining all system based results
    # sys_online_info_df_all = pd.concat(dfs_sys, axis='rows', ignore_index=True)
    sys_online_info_df_all = pd.DataFrame(pt_sys_all)
    sys_online_info_df_all['time_mpInit_total_sec'] = pt_mp_init_total
    sys_online_info_df_all['time_mpClose_total_sec'] = pt_mp_close_total
    sys_online_info_df_all['time_disagg_total_sec'] = pt_disagg_total
    
    return  p_info_df_all, sys_online_info_df_all

def ParallelProcessing(Np, dw, Nh_chunk, house_ids):
    """running the NILM algorithm in multiprocess"""
    
    house_ids = house_ids[0:Nh_chunk]
    Nh = len(house_ids)
    dfs_process =[]
    pt_sys_all = []
    onlineId = 0
    pt_disagg_total = 0
    pt_mp_init_total = 0
    pt_mp_close_total = 0
    start_datetimes_online = start_datetimes_online_all

    t_start = time.time()

    while len(house_ids) > 0:  
        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(onlinedw_d)

        t_start_online = time.time()
        onlineId = onlineId + 1
        Nh_online = len(house_ids)
        pt_sys_dict = {
                        "numProcesses":Np, "dw_min":dw, 
                        "num_houses":Nh, "num_houses_chunk":Nh_chunk, "num_houses_online":Nh_online, 
                        "onlineId":onlineId
                    }

        t_start = time.time()
        pool = mp.Pool(processes=Np)
        pt_mp_init = time.time() - t_start
        
        # pt_sys_init = getSystemLevelInfo()

        t_start = time.time()
        results = [pool.apply_async(AlgorithmMonitor, args=(Np, dw, Nh, Nh_chunk, Nh_online, onlineId, house_id, house_ids_all_org[house_id-1], start_datetimes_online[house_id][0], )) 
                            for house_id in house_ids]

        [p.wait() for p in results if not p.ready()] 
        output = [p.get() for p in results] 
        dfs_process.append(pd.DataFrame(output))

        pt_disagg_online = time.time() - t_start
       
        t_start = time.time()
        assert not pool._cache, 'cache = %r' % pool._cache
        
        # closing multiprocessing pool
        for worker in pool._pool:
            assert worker.is_alive()
        pool.close()
        pool.join()
        
        pt_mp_close = time.time() - t_start

        pt_mp_init_total = pt_mp_init_total + pt_mp_init
        pt_disagg_total = pt_disagg_total + pt_disagg_online
        pt_mp_close_total = pt_mp_close_total + pt_mp_close

        # pt_sys_final = getSystemLevelInfo()
        # pt_sys_d = [pt_sys_final[s] - pt_sys_init[s] for s in range(len(pt_sys_init))]

        # additional code for creating the dw slot groups, simultes db update for the next reading
        house_ids, start_datetimes_online = GetNextOnlineData(house_ids, start_datetimes_online, dataset, dw)
        
        # pt_sys.append(pt_online)

        pt_sys_dict["time_mpInit_sec"] = pt_mp_init
        pt_sys_dict["time_disagg_online_sec"] = pt_disagg_online
        pt_sys_dict["time_mpClose_sec"] = pt_mp_close
        # pt_sys.extend(pt_sys_d)
        # pt_sys.extend(pt_sys_final)

        pt_online = time.time() - t_start_online
        pt_sys_dict["time_total_online_sec"] = pt_online

        pt_sys_all.append(pt_sys_dict)     

        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(onlinedw_d)

    #  Combining all process based results
    p_info_df_all = pd.concat(dfs_process, axis='rows', ignore_index=True)
    p_info_df_all['time_disagg_total_sec'] = pt_disagg_total

    # Combining all system based results
    # sys_online_info_df_all = pd.concat(dfs_sys, axis='rows', ignore_index=True)
    sys_online_info_df_all = pd.DataFrame(pt_sys_all)
    sys_online_info_df_all['time_mpInit_total_sec'] = pt_mp_init_total
    sys_online_info_df_all['time_mpClose_total_sec'] = pt_mp_close_total
    sys_online_info_df_all['time_disagg_total_sec'] = pt_disagg_total
    
    return  p_info_df_all, sys_online_info_df_all

def SaveExperimentSettings(costlog_path, exprmnt_time_min):
    
    exprmnt_setting_dict = OrderedDict()
    exprmnt_setting_dict['Nh_chunks'] =  list(Nh_chunks)
    exprmnt_setting_dict['Nps'] =  list(Nps)  
    exprmnt_setting_dict['num_Nh_chunks'] =  len(Nh_chunks)  
    exprmnt_setting_dict['num_Nps'] =  len(Nps) 
    exprmnt_setting_dict['num_houseload_duplication'] =  num_houseload_duplication 
    exprmnt_setting_dict['isdays'] =  isdays
    exprmnt_setting_dict['disaggtimewindow'] =  disaggtimewindow
    exprmnt_setting_dict['isoneonline'] =  isoneonline
    exprmnt_setting_dict['num_repeated_expt'] =  num_repeated_expt 
    exprmnt_setting_dict['init_d'] = init_d 
    exprmnt_setting_dict['exprmnt_d'] = exprmnt_d 
    exprmnt_setting_dict['Nhchunk_d'] = Nhchunk_d 
    exprmnt_setting_dict['Np_d'] = Np_d 
    exprmnt_setting_dict['onlinedw_d'] = onlinedw_d
    exprmnt_setting_dict['cost_monitoring_run_time (minute)'] =  exprmnt_time_min

    filename = "{}/{}".format(costlog_path, setting_dict["meter_log"]["setting_file"])
    util.SaveJson(filename, exprmnt_setting_dict,)
    util.PrintDict(exprmnt_setting_dict)
    
@util.timer
def RunExperiments(system_log_path, POs, Nh_chunks, Nps, dw, expr_id=None):
    """
    Executes the target NILM algorithm for a given list of load scales and multiprocessing selections hierarchical. 
    For the given experiment scenarios specified in the setting.JSON such as list of number of houses, list of number of processes, the algorithm monitoring module executes the target NILM algorithm encapsulated with SyncPatternInsertion() in a nested loops for each unique experiment combinations.
    """
    
    for PO in POs:
        print("ProcessingOptions:", PO)
        if PO=='SP':
            for Nh_chunk in Nh_chunks:
                print("NumberHousePerMp:", Nh_chunk)
                house_ids = np.random.choice(house_ids_all, Nh_chunk, replace=False)
                Np = 1
                print("Parallel Processing Np:", Np)
                
                # add synch patten in the power consumption measurement 
                SyncPatternInsertion(Np_d)       

                p_info_df_all, sys_online_info_df_all = SerialProcessing(Np, dw, Nh_chunk, house_ids)  

                filename = util.resource_log_filename_format.format(system_log_path, PO, dw, expr_id, Nh_chunk, Np)
                util.SaveDatatoCSV("{}_per_process.csv".format(filename), p_info_df_all)
                util.SaveDatatoCSV("{}_per_sys.csv".format(filename), sys_online_info_df_all)
            
                # p_info_df_all, sys_online_info_df_all = None, None
                del p_info_df_all, sys_online_info_df_all
                # gc.collect()
                
                # add synch patten in the power consumption measurement 
                SyncPatternInsertion(Np_d)   
                                         
        elif PO == 'MP':             
            for Nh_chunk in Nh_chunks:
                print("NumberHousePerMp:", Nh_chunk)
                house_ids = np.random.choice(house_ids_all, Nh_chunk, replace=False)
                
                # add synch patten in the power consumption measurement 
                SyncPatternInsertion(Nhchunk_d)
                
                for Np in Nps:    
                    print("Parallel Processing Np:", Np) 

                    # add synch patten in the power consumption measurement 
                    SyncPatternInsertion(Np_d)       

                    p_info_df_all, sys_online_info_df_all = ParallelProcessing(Np, dw, Nh_chunk, house_ids)

                    filename = util.resource_log_filename_format.format(system_log_path, PO, dw, expr_id, Nh_chunk, Np)
                    util.SaveDatatoCSV("{}_per_process.csv". format(filename), p_info_df_all)
                    util.SaveDatatoCSV("{}_per_sys.csv".format(filename), sys_online_info_df_all)
                    # p_info_df_all, sys_online_info_df_all = None, None
                    del p_info_df_all, sys_online_info_df_all
                    # gc.collect()      
                    
                    # add synch patten in the power consumption measurement  
                    SyncPatternInsertion(Np_d) 
                
                # add synch patten in the power consumption measurement       
                SyncPatternInsertion(Nhchunk_d)  

def main(*arg, **params):
    """
    Provides the core functionality of the computational cost measurement of the NILM algorithm. 
    Using the experiment setting and generated large scale load data, it calls the RunExperiments to monitor the associated costs. 
    """
    # removing previous old NILM result to keep database exploding with repeated experimentations
    nilmAlg.CleanOldResultLogs()

    print("DisagWindow:", disaggtimewindow)
    print("Number of Process:", Nps)
    print("Number of Houses:", Nh_chunks)

    POs = ['SP', 'MP']
    POs = POs[1:] # currently focus only on multiprocessing
    print("Processing: {}".format(POs))

    # add synch patten in the power consumption measurement 
    SyncPatternInsertion(init_d)
    
    for expr_id in range(0, num_repeated_expt):
        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(exprmnt_d)    
        
        # call cost montioring experiment   
        RunExperiments(system_log_path, POs, Nh_chunks, Nps, disaggtimewindow, expr_id)
        
        # add synch patten in the power consumption measurement 
        SyncPatternInsertion(exprmnt_d)  
         
    # add synch patten in the power consumption measurement 
    SyncPatternInsertion(init_d)  

if __name__ == '__main__': 
    
    gc.collect()
    util.clear()

    #  creating working directories in the target result folder
    costlog_path, system_log_path, meter_log_path = CreateWorkingDirs(result_path)

    t_start = time.time()

    # call the cost monitoring function
    main()

    exprmnt_time_min = np.round((time.time() - t_start)/60, 3)

    # save cost monitoring settings
    SaveExperimentSettings(costlog_path, exprmnt_time_min)
    
    print("total all experiment time (min): ", exprmnt_time_min) 

    msg = "{}\n The result of the cost monitoring can be found in {}. \n Please export export meter reading into {} and save it as {}.\n{}"\
        .format("*"*60, costlog_path, setting_dict["meter_log"]["log_dir"], setting_dict["meter_log"]["log_file"], "*"*60)

    print(msg)
