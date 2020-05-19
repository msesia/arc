

from run_experiment_real_data import run_experiment

alpha = 0.1

DATASET_LIST   = ["mice", "fashion", "mnist", "cifar10"]
N_TRAIN_LIST = [500, 1000, 5000, 10000]

# Data sets directory 
dataset_base_path = '~/mydata/classification_data/'

# Where to write results
out_dir = "./results"

for EXP_id in range(100):
    for N_TRAIN_LIST_id in range(4):
       for DATASET_LIST_id in range(4):
 
           dataset_name = DATASET_LIST[DATASET_LIST_id]
           n_train = N_TRAIN_LIST[N_TRAIN_LIST_id]
           
           run_experiment(out_dir,
                          dataset_name,
                          dataset_base_path,
                          n_train,
                          alpha=alpha,
                          experiment=EXP_id)
