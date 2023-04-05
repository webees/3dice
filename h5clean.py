import os
import time

dir = os.path.join(os.getcwd(), 's1')
print(dir)

# max_files = 51

while True:
    try:
        time.sleep(1)
        file_list = os.listdir(dir)
        h5_count = len([f for f in file_list if f.endswith('.h5')])
        print("H5_COUNT:", h5_count)
        # if h5_count < max_files:
        #     continue
        min_vacc = float("inf")
        min_file = ""
        for file in file_list:
            if file.endswith(".h5"):
                vacc_str = file.split("_vacc")[1].replace('.h5', '')
                vacc = float(vacc_str)
                if vacc < 0.6:
                    min_vacc = vacc
                    min_file = file
        if min_file != "":
            os.remove(os.path.join(dir, min_file))
            print(f"❌ {min_file}")
    except PermissionError:
        time.sleep(1)
