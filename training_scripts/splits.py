import shutil

import pathlib
from sklearn.model_selection import train_test_split
from os.path import join
import os

seed=11

theorems_path = "proven_theorems/"
data_path = "train_data/"

files = [str(f) for f in pathlib.Path(theorems_path).rglob('*.json')]

train, other = train_test_split(files, test_size=0.05, random_state=seed)
valid, test = train_test_split(other, test_size=0.5, random_state=seed)


shutil.rmtree(data_path, ignore_errors=True)

for f in train:
    os.makedirs("/".join(join(data_path + "train", f[len(theorems_path):]).split("/")[:-1]), exist_ok=True)
    shutil.copy(f, join(data_path + "train", f[len(theorems_path):]))

for f in valid:
    os.makedirs("/".join(join(data_path + "validation", f[len(theorems_path):]).split("/")[:-1]), exist_ok=True)
    shutil.copy(f, join(data_path + "validation", f[len(theorems_path):]))

for f in test:
    os.makedirs("/".join(join(data_path + "test", f[len(theorems_path):]).split("/")[:-1]), exist_ok=True)
    shutil.copy(f, join(data_path + "test", f[len(theorems_path):]))
