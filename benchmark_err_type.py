########################################
# Benchmark
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# November 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################
# Revised by Wei Ni

########################################
import sys
sys.path.append('./raha-master')
import time
import pandas as pd
import raha
from measure import measure_detect
########################################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

########################################
class Benchmark:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.RUN_COUNT = 1
        # self.DATASETS = ["hospital", "flights", "beers", "billionaire", "rayyan", "movies"]
        self.DATASETS = ["movies"]
        self.SYSTEMS = ["Raha"]
        # self.err_type = ['typos', 'missing_values', 'pattern_violations', 'rule_violations', 'outliers']
        self.err_type = ['typos', 'missing_values', 'pattern_violations', 'rule_violations', 'outliers', 'mixed_err'] 

    def experiment_1(self):
        """
        This method conducts experiment 1.
        """
        import os
        import shutil
        
        # Delete all raha-baran* folders in ./data directory
        data_dir = './data'
        print("------------------------------------------------------------------------\n"
              "-----------------Experiment 1: Comparison with Baselines----------------\n"
              "------------------------------------------------------------------------")
        # results = {sas: {dn: [] for dn in self.DATASETS} for sas in stand_alone_systems}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            detector.LABELING_BUDGET = 2
            competitor = raha.baselines.Baselines()
            for err_type in self.err_type:
                print(f"*****Running {err_type} experiment*****")
                for item in os.listdir(data_dir):
                    if item.startswith('raha-baran') and os.path.isdir(os.path.join(data_dir, item)):
                        shutil.rmtree(os.path.join(data_dir, item))
                dataset_name = 'beers'
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": './data/' + f'{dataset_name}_clean_{err_type}.csv',
                    "clean_path": './data/' + dataset_name + '_clean.csv'
                }
                # d = raha.dataset.Dataset(dataset_dictionary)
                dirty_csv = pd.read_csv(dataset_dictionary['path'])
                attr_list = list(dirty_csv.columns)
                for stand_alone_system in self.SYSTEMS[::-1]:
                    start_time = time.time()
                    if stand_alone_system == "dBoost":
                        detection_dictionary = competitor.run_dboost(dataset_dictionary)
                    elif stand_alone_system == "NADEEF":
                        detection_dictionary = competitor.run_nadeef(dataset_dictionary)
                    elif stand_alone_system == "KATARA":
                        detection_dictionary = competitor.run_katara(dataset_dictionary)
                    elif stand_alone_system == "ActiveClean":
                        detection_dictionary = competitor.run_activeclean(dataset_dictionary)
                    else:
                        detection_dictionary = detector.run(dataset_dictionary)
                    # er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    end_time = time.time()
                    exec_time = end_time - start_time
                    # print(detection_dictionary)
                    print(f"{stand_alone_system} took {exec_time} seconds to detect {dataset_name}.")
                    # results[stand_alone_system][dataset_name].append(er)
                    detection_list = []  # dic_2_list
                    for (row_index, col_index), value in detection_dictionary.items():
                        detection_list.append([int(row_index), attr_list[col_index]])
                    detection_list = sorted(detection_list, key=lambda x: x[0])  # rerank
                    # print(detection_list)
                    measure_detect(dataset_dictionary['clean_path'], dataset_dictionary['path'], detection_list,
                                   './result/baselines/' + dataset_name + f'_{err_type}_' + stand_alone_system + '_det_res.txt', err_type)
                    # with open('./result/detect_baselines_new/' + dataset_name + '_' + stand_alone_system + '_detect_list.txt', 'w') as file:
                    #     for item in detection_list:
                    #         line = ', '.join(str(x) for x in item)
                    #         file.write(line + '\n')
########################################


########################################
if __name__ == "__main__":
    app = Benchmark()
    app.experiment_1()