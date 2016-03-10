
from multiprocessing import Pool, Process
from settings import *
import os

from byte_code_extraction_facade import byte_extraction
from asm_extraction_facade import asm_extraction
from feature_selection_facade import feature_fusion
from classification_facade import classification


def byte_code_worker(datasets):
    byte_pool = Pool(2)
    byte_pool.map(byte_extraction, datasets)


def asm_code_worker(datasets):
    byte_pool = Pool(2)
    byte_pool.map(asm_extraction, datasets)


def main():
    steps = ['feature extraction', 'feature fusion', 'classification']
    step = steps[2]
    datasets = ['train', 'test']

    if step == 'feature extraction':
        print('Feature Extraction STEP')
        print('=======================')
        for dataset in datasets:
            if not os.path.isdir(SAVED_PATH_CSV + dataset):
                os.makedirs(SAVED_PATH_CSV + dataset)

        p1 = Process(target=byte_code_worker, args=(datasets,))
        p2 = Process(target=asm_code_worker, args=(datasets,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    elif step == 'feature fusion':
        print('Feature Fusion STEP')
        print('===================')
        if not os.path.isdir(SAVED_PATH_CSV + 'train'):
            print "There is no train folder!!!"
            return
        if not os.path.isdir(COMBINED_PATH_CSV):
            os.makedirs(COMBINED_PATH_CSV)
        feature_fusion()

    elif step == 'classification':
        print('Classification STEP')
        print('===================')
        substeps = ['cv', 'test']
        # if substep equals to 0, the program runs cross validation, otherwise it runs on the test dataset
        substep = 0
        if substeps[substep] == 'cv':
            print('Testing with Cross validation')
            classification(TRAIN_FILE, select = False, bagging = False, test = 'cv')
        else:
            print('Testing on the test dataset')
            classification(TRAIN_FILE, bagging = True, test = TEST_FILE)


if __name__ == '__main__':
    main()