import os
import time

__PROJECT = 'm5'
__DIR = os.path.join(os.getcwd(), __PROJECT)

MAX_FILES = 20
MIN_VACC = 0.6

while True:
    try:
        time.sleep(1)
        file_list = os.listdir(__DIR)
        h5_count = len([f for f in file_list if f.endswith('.h5')])
        print("H5_COUNT:", h5_count)
        if h5_count <= MAX_FILES:
            continue
        min_vacc = float("inf")
        min_file = ""
        for file in file_list:
            if file.endswith(".h5"):
                vacc_str = file.split("_vacc")[1].replace('.h5', '')
                vacc = float(vacc_str)
                if vacc < min_vacc and vacc < MIN_VACC:
                    min_vacc = vacc
                    min_file = file
        if min_file != "":
            os.remove(os.path.join(__DIR, min_file))
            print(f"❌ {min_file}")
    except PermissionError:
        time.sleep(1)
