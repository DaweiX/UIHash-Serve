import hashlib
from os import rename, mkdir
from os.path import join, exists, basename, dirname, abspath
import platform
import subprocess

from tqdm import tqdm
import yaml

from .ui_crawler import UICrawler


# load configs
home_path = dirname(dirname(abspath(__file__)))
with open(join(home_path, "config.yml"), 'r') as f:
    configs = yaml.safe_load(f)

app_folder = configs["path"]["APK"]
uihash_data = configs["path"]["DATA"]
log_file = configs["log"]["COLLECT"]
log_level = configs["log"]["LEVEL"]


def rename_file(apk_file, chunk_size):
    sha256 = hashlib.sha256()
    with open(apk_file, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    apk_sha256 = sha256.hexdigest().upper()
    dname, fname = dirname(apk_file), basename(apk_file)
    new_fname = f"{apk_sha256}.{fname.split('.')[-1]}"
    rename(apk_file, join(dname, new_fname))
    return new_fname


def test_apks(chunk_size=8192):
    from os import listdir
    apks = listdir(app_folder)
    for apk in tqdm(apks, desc="Process apks"):
        _ = rename_and_test_apk(apk, chunk_size)


def rename_and_test_apk(apk: str, chunk_size: int=8192) -> str:
    apk_file = join(app_folder, apk)
    if not exists(apk_file):
        print(f"File not found: {apk_file}")
        exit(-1)
        
    apk_new_name = rename_file(apk_file, chunk_size)
    return test_apk(apk_new_name)


def test_apk(apk: str) -> str:
    # TODO: we take an apk file from ${app_folder} as an input
    # for online usage, we need a file listener, when a 
    # new apk is stored in ${app_folder}, collect its ui

    output_path = join(uihash_data, ''.join(apk.split('.')[:-1]))
    if not exists(output_path):
        mkdir(output_path)
    else:
        return output_path
    
    if platform.system() == "Windows":
        cmd = ["taskkill", "/f", "/t", "/im", "adb"]
    else:
        cmd = ["killall", "adb"]
    subprocess.run(cmd)
    uc = UICrawler(ip="localhost:5555", log_level=log_level, log_path=log_file,
                   input_apk=join(app_folder, apk), output_path=output_path, safe_package="")
    uc.dump_apk()
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        test_apks()
    else:
        rename_and_test_apk(sys.argv[1])