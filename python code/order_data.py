import shutil
import os
import glob
from tqdm import tqdm

for filename in tqdm(os.listdir('data/printed_digits')):
    number = filename.split('_')[2].replace('.png', '')
    shutil.copy2(f'data/printed_digits/{filename}', f'data/numbers/{number}/{filename}')