# COMMIT-NILM: COMputational MonItoring Tool for NILM Algorithms

COMMIT-NILM provides monitoring, data-driven modeling and analysis for cloud-based NILM (Non-intrusive Load Monitoring) algorithm's computational costs such as processing time and energy consumption from system-level power measurements.

## Instruction

To utilize the COMMIT, a user have to appropriately set the power meter and the target NILM integration setups first.  
1. Generate the baseline house load dataset from source datasets such as public datasets using Load Data Generator. This prepares a CSV file, *setting["cleaneddata\_db\_ref\_file"]*, with cleaned a data in daily chunks:
        
        python LoadDataGenerator.py
        
2. Connect the power supply to the computing system or server through the meter in between. The meter also connects to the server via USB serial connection for configuration management and logged data retrieval. 
3. Place the code implementation of the target NILM algorithm in a dedicated folder, */code/NILMAlgorithm/* and configure the *TargetAlgNILMRunner.py* according to the algorithm to make it accessible from the monitoring module. 
4. To run the cost monitoring experiments initially, set the experiment parameters such as the minimum and maximum number of houses and parallel processors, number of experiment repetitions, and the disaggregation time-window  size of the NILM algorithm. All the configuration and settings are done in a setting file, named as *setting.JSON*. 
5. Start the cost monitoring:

        python CostMonitor.py
 
6. After the power monitoring has finished, power log is exported from the meter's memory into an external CSV file, defined in *setting["log_file]*. 
7. The regions of interest (ROIs) that belong to the executions of the NILM algorithm are filtered and extracted from the exported monitored power log using ROI Extraction:
  
        python MeterReadingROIExtractor.py
         
8. The preparation of the cost datasets from the extracted ROIs follows: 

        python CostDataPreparation.py

9. Processing time and energy costs are modeled using statistical and machine learnig models: 

        python CostModeler.py

10. Finally, cost analysis and prediction are carried-out: 

        python CostAnalyzer.py
      
Moreover, if Wattsup? Pro or similar meter is going to be used for monitoring the server power, configure the meter to the following recommended settings: 

        * data storage location: meter's internal memory, overwrite during memory full: false, sampling time: 1 second and parameters to record: power in Watts. 
        
        * Just before running the cost monitoring starts, the memory of the meter needs to be cleared manually, and the meter automatically starts recording the server's power consumption.

** Step 1 to 4 are one time operations, while 5-10 need to be carried-out for every experiment setting changes defined in 5**

** Generally, setting multiprocessing beyond the number of logical processor of system is not recommended due to the limited computation leverage from high context switching though it is possible!**

![Large Scale HouseLoad Data Generation](https://github.com/muleina/COMMIT-NILM/blob/master/COMMIT-NILM_instruction_howtouse.png)

## Sample Example: Cost Monitoring of Online-NILM Algorithm
Here are sample notebooks and screenshots from command line when COMMIT-NILM is applied to monitor computational cost of our *Cloud-based Online-NILM* algorithm. 

### COMMIT-NILM using Jupyter-Notebooks
* [Large Scale HouseLoad Data Generation](https://github.com/muleina/COMMIT-NILM/blob/master/code/notebooks/COMMIT-NILM_prepare_largescale_houseloaddata_dbref_example.ipynb)

* [Cost Modeling](https://github.com/muleina/COMMIT-NILM/blob/master/code/notebooks/COMMIT-NILM_cost_modeling_example.ipynb)

* [Cost Analysis](https://github.com/muleina/COMMIT-NILM/blob/master/code/notebooks/COMMIT-NILM_cost_analysis_example.ipynb)

### COMMIT-NILM using CLI
* [Wattsup? Pro Meter Reading ROI Extraction and Cost Dataset Preparation](https://github.com/muleina/COMMIT-NILM/blob/master/code/results/System_CVS/costlog_2/README.md)

* [Cost Modeling](https://github.com/muleina/COMMIT-NILM/blob/master/code/results/System_CVS/modeling/README.md)

* [Cost Analysis](https://github.com/muleina/COMMIT-NILM/blob/master/code/results/System_CVS/analysis/README.md)

## Requirement
COMMIT-NILM runs on Python 3.0+. The load data generation and cost monitoring modules also runs on Python 2.7 to provide backward compatibility for NILM algorithm written in older versions.

Some of the dependencies are python 3, numpy, pandas, scikit-learn, scipy, psutil, seaborn, matplotlib, json and tqdm.

        pip install -r requirements.txt
        
## Papers: 
1. [Computational cost analysis and data-driven predictive modeling of cloud-based online NILM algorithm](https://ieeexplore.ieee.org/abstract/document/9325000) 
        
        @article{asres2021computational,
          title={Computational cost analysis and data-driven predictive modeling of cloud-based online NILM algorithm},
          author={Asres, Mulugeta Weldezgina and Ardito, Luca and Patti, Edoardo},
          journal={IEEE Transactions on Cloud Computing},
          year={2021},
          publisher={IEEE}
        }

