import os, re, fnmatch, pickle
import numpy as np
from os.path import isfile, join
import pandas as pd
import hashlib
import shutil


class io:
    @staticmethod
    def change_to_current_directory():
        current_path = os.getcwd()
        os.chdir(current_path)
        return current_path

    @staticmethod
    def get_current_directory():
        current_path = os.getcwd()
        return current_path

    @staticmethod
    def change_to_home_directory():
        home_path = os.path.expanduser('~')
        os.chdir(home_path)
        return home_path

    @staticmethod
    def read_all_lines(file_name):
        with open(file_name) as f:
            content = f.readlines()
            return content

    @staticmethod
    def get_files_in_directory(directory_path, file_extension='*', subdirectory=True):
        files = []
        if subdirectory == True:
            if file_extension == '*':
                files = [f
                         for dirpath, dirnames, files in os.walk(directory_path)
                         for f in files
                         if isfile(os.path.join(dirpath, f))
                         ]
                return files
            else:
                files = [os.path.join(dirpath, f)
                         for dirpath, dirnames, files in os.walk(directory_path)
                         for f in fnmatch.filter(files, '*' + file_extension)
                         ]
                return files
        else:
            if file_extension == '*':
                files = [f for f in os.listdir(directory_path)
                         if isfile(join(directory_path, f))]
                return files
            else:
                files = [f for f in os.listdir(directory_path)
                         if f.endswith('.' + file_extension)
                         if isfile(join(directory_path, f))
                         ]
                return files

    @staticmethod
    def get_md5_file_name(file_path):
        file_name = os.path.basename(file_path)
        return re.findall(r"([a-fA-F\d]{32})", file_name)[0]

    @staticmethod
    def load_pickle(file_path):
        rows = []
        file_object = open(file_path, 'r')
        while (1):
            try:
                rows.append(pickle.load(file_object))
            except (EOFError, pickle.UnpicklingError):

                break
        file_object.close()
        return rows

    @staticmethod
    def get_md5_family(list_of_objects):
        md5_family_dict = {}
        for item in list_of_objects:
            md5_family_dict[item[0]] = item[1]
        return md5_family_dict

    @staticmethod
    def save_as_csv(Y, X_raw, glob_f, path):
        glob_f = glob_f + ['Class']

        newY = np.zeros(len(Y))
        uniqueY = np.unique(Y)
        # print(uniqueY)
        i = 1
        for k in range(len(Y)):
            index = np.where(uniqueY == Y[k])
            newY[k] = index[0] + 1
        result = np.column_stack((X_raw, newY))
        result = np.row_stack((glob_f, result))
        np.savetxt(path, result, delimiter=',', fmt="%s")


    @staticmethod
    def save_samples_as_csv(samples):
        for sample in samples:
            pass


    @staticmethod
    def save_objects_as_pickle(objects, file_name):
        with open(file_name, "wb") as f:
            for object in objects:
                pickle.dump(object, f)
        f.close()


    @staticmethod
    def load_from_csv(modes_files_path, mode):
        mode_file_path = modes_files_path + '/' + mode + '.csv'
        mode_file = pd.read_csv(mode_file_path, delimiter=',')

        mode_feature_vectors = mode_file.ix[:, :-1]
        mode_class_labels = mode_file.ix[:, -1]

        return mode_feature_vectors, mode_class_labels

    @staticmethod
    def is_file(file_name):
        return os.path.isfile(file_name)

    @staticmethod
    def get_directories(directory_path):
        return os.listdir(directory_path)

    @staticmethod
    def get_upper_directory_name(file_path):
        par = os.path.dirname(file_path)
        return par[par.rfind('/')+1:]

    @staticmethod
    def get_md5(file_path):
        hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        return hash.hexdigest()

    @staticmethod
    def remove_file(file_path):
        os.remove(file_path)

    @staticmethod
    def move_file(src, dst):
        shutil.move(src, dst)

    @staticmethod
    def get_file_name(file_path):
        return os.path.basename(file_path)

    @staticmethod
    def save_txt(strings, file_path, append= True):
        if not os.path.isfile(file_path):
            file = open(file_path, 'w')
            file.close()
        with open(file_path, "a") as myfile:
            for string in strings:
                myfile.write(string+'\n')
            myfile.close()

