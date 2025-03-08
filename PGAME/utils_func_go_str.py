import csv
import pandas as pd
from statistics import mode
from setting_AI import *

csv_path = "dataCSV/direction_control.csv"
csv_mask_path = "dataCSV/direction_control_mask.csv"
csv_straight_path = "dataCSV/direction_straight.csv"
csv_back_control_path = "dataCSV/back_control.csv"



def ADD_DATA_CSV_MASK_DIRECTION(direction, angle):
    with open(csv_mask_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([direction, angle])
    
    data_csv = pd.read_csv(csv_mask_path)
    
    if len(data_csv) == 10000:
        file_start = pd.read_csv(csv_mask_path, nrows=0)
        file_start_new = pd.DataFrame(columns=file_start.columns)
        file_start_new.to_csv(csv_mask_path, index=False)

def ADD_DATA_CSV_DIRECTION(direction, angle):
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([direction, angle])
        
def ADD_DATA_CSV_DIRECTION_STRAIGHT(direction, angle):
    with open(csv_straight_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([direction, angle])
    
    data_csv = pd.read_csv(csv_straight_path)
    if len(data_csv) == 500:
        CLEAN_DATA_CSV_DIRECTION_STRAIGHT()

def CLEAN_DATA_CSV_DIRECTION():
    # Clear "direction_control.csv"
    file_start = pd.read_csv(csv_path, nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv(csv_path, index=False)

    # Clear "direction_control_mask.csv"
    file_start = pd.read_csv(csv_mask_path, nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv(csv_mask_path, index=False)

def ADD_DATA_CSV_BACK_CONTROL(direction, angle):
    with open(csv_back_control_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([direction, angle])
    
def CLEAN_DATA_CSV_BACK_CONTROL():
    # Clear "back_control.csv"
    file_start = pd.read_csv(csv_back_control_path, nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv(csv_back_control_path, index=False)
    
def CLEAN_DATA_CSV_DIRECTION_STRAIGHT():
    # Clear "direction_control.csv"
    file_start = pd.read_csv(csv_straight_path, nrows=0)
    file_start_new = pd.DataFrame(columns=file_start.columns)
    file_start_new.to_csv(csv_straight_path, index=False)
    
def BOTTOM_DATA_CSV_CHECK():
    data_csv_ = pd.read_csv(csv_path)
    last_row = data_csv_.iloc[-1]
    return (last_row["direction"], last_row["angle"])
    
def CHECK_PUSH():
    push_variable = None
    dr_back, an_back = None, None
    data_csv_ = pd.read_csv(csv_mask_path)
    direction_list_to_mode = list(data_csv_['direction'][-count_control:])
    if len(direction_list_to_mode) > 0:
        direction_mode = mode(direction_list_to_mode)
        max_angle = max(list(data_csv_['angle'][:count_control]))
        if len(pd.read_csv(csv_path)) == 0:
            dr_back, an_back = direction_mode, max_angle
            ADD_DATA_CSV_DIRECTION(direction_mode, max_angle)
            # ADD_DATA_CSV_BACK_CONTROL(direction_mode, max_angle)
            return f"{direction_mode}:{max_angle:03d}", dr_back, an_back
        else:
            bottom_data_csv_check = BOTTOM_DATA_CSV_CHECK()
            if bottom_data_csv_check[0] != direction_mode or (abs(bottom_data_csv_check[1] - max_angle) >= threshold_scale):
                CLEAN_DATA_CSV_DIRECTION()
                # ADD_DATA_CSV_DIRECTION(direction_mode, max_angle)
                dr_back, an_back = direction_mode, max_angle
                return f"{direction_mode}:{max_angle:03d}", dr_back, an_back
            else:
                return push_variable, dr_back, an_back
            
    return push_variable, dr_back, an_back
    
    

        
    