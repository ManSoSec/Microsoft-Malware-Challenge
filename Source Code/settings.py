from handle_io import io

# This folder should contain 2 folders, namely train_gz and test_gz, containing gzip of samples
DATASET_PATH = io.get_current_directory()[0:io.get_current_directory().rindex('/')] + '/Dataset/'

SAVED_PATH_CSV = DATASET_PATH + 'FeatureCategories/'
TRAIN_ID_PATH = DATASET_PATH +'trainLabels.csv'
TEST_ID_PATH = DATASET_PATH +'sorted_test_id.csv'
COMBINED_PATH_CSV = DATASET_PATH + 'combination/'
BYTE_TIME_PATH = DATASET_PATH + 'byte_time.txt'
ASM_TIME_PATH = DATASET_PATH + 'asm_time.txt'
APIS_PATH = DATASET_PATH + 'APIs.txt'
TRAIN_FILE = DATASET_PATH + 'train' + '/LargeTrain.csv'
TEST_FILE = DATASET_PATH + 'test' + '/LargeTest.csv'
SUB_PATH = DATASET_PATH + 'submissions/'

