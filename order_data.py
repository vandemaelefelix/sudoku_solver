import shutil
import os
import glob
from tqdm import tqdm

for filename in tqdm(os.listdir('data/')):
    number = filename.split('_')[2].replace('.png', '')
    shutil.copy2(f'data/{filename}', f'data_ordered/{number}/{filename}')