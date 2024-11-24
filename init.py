from os import listdir
from os.path import dirname, abspath, join

import yaml
from tqdm import tqdm

from ui2hash.main import apk2data
from ui2hash.database import insert_data
    

# load configs
home_path = dirname(abspath(__file__))
with open(join(home_path, "config.yml"), 'r') as f:
    configs = yaml.safe_load(f)

# some constant
DATA_PATH = configs["path"]["DATA"]

if __name__ == "__main__":
    apks = listdir(DATA_PATH)
    c = 0
    for apk in tqdm(apks, desc="feed data"):
        c += len(listdir(join(DATA_PATH, apk)))
        results = apk2data(join(DATA_PATH, apk))
        for result in results:
            insert_data(*result)

    print(f"Total UI: {c/2}")