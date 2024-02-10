# Enefit-Predict-Energy-Behavior-of-Prosumers

# Target 🎯

In our project, there was no target transformation. The target was predicted as is. An attempt was made to predict the difference between the target and the average values over several days, relative to some other variables, but the result worsened.

## Features 🛠️

We used a standard set of features. Additionally, we used lagged weather predictions for Production, with one and two-hour intervals. For Consumption, instead of that, we used 7-day lags for temperature (predictions and historical data).

## Ensembling 🤝

We used 6 LGBMs in ensembling for Consumption and 6 LGBMs for Production.

## Re-training 🔄

Every day, we re-train all LGB models on GPUs. We used child processes and distributed the load on two GPUs on Kaggle T4.

## Sampling 🎲

For each model in the ensembles separately, we randomly removed a portion of the data to speed up the training process. The first 100 days were left untouched. Then, we kept different ranges of data from 0.5 to 0.2 of the total data. Something like dropout. An addition to our team lead =))) I want to thank my team ds60 (Mikhail, Pavel) for their selfless struggle kaggle community, for ideas!😀

A little bit of essence😁

## What didn't fit:

1) An attempt to dilute the ensemble of LGBMRegressor, CatBoost, and XGb models made it heavier and failed to pass at 9 o'clock😭

2) Attempts to change the ensemble to StackingRegressor and BaggingRegression, to create a decisive model for the ensemble, also failed😔

3) The option is divided into regions and create a double model for each and then assemble everything into one, it also failed in time.🥺

## What came up:🚀

- A sign of working days
- Calculation of ACF and PACF for signs 

![ACF and PACF Calculation 1](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13372827%2Fb30ee7434e24a2385fe73e78ee354a35%2F__results___22_0.png?generation=1706931601517243&alt=media)

![ACF and PACF Calculation 2](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13372827%2F5cf809671cdff8793a932a7c73ab8318%2F__results___23_0.png?generation=1706931617739602&alt=media)

- Using parallelization in downloading 2 maps using the new LGBM library


```
# # For the card on the T4x2 cable
!pip uninstall -y lightgbm
!pip install /kaggle/input/lightgbm420-cuda/lightgbm-4.2.0-py3-none-manylinux_2_35_x86_64.whl
import lightgbm as lgb

num_processes = 2
# number of parallel threads to train each model
n_jobs = None
# Type of accelerator in the system: gpu or cuda
gpu_type = 'cuda'
# the number of GPUs in the system
gpus_n = 2
```
- Block selection
```
# For local computing. The next data_block_id of the deleted selection
# # The initial information from the next data_block_id and up to the end will contain the text
# train_end_data_block_id = 517
# train_end_data_block_id = 600+360
# train_end_data_block_id = 600
train_end_data_block_id = 598
#train_end_data_block_id = 636
# train_end_data_block_id = 680
```
- and
```
# Increases the dataframe to May 2024
# df is the dataframe that we are increasing
# date_col is the column where we cut out the old data
# columns with data to which we add 365 days
def enlarge_df(df_name, date_col, add_date_cols):

    print('Increasing the size for:', df_name)
    df = pd.read_csv(os.path.join(root, df_name))
    for col in add_date_cols:
        df[col] = pd.to_datetime(df[col])
    
    start_date = df[date_col].max() +pd.DateOffset(days=1)
# The end date by which to increase the dataframe    
    end_date = pd.to_datetime('2024-05-01')
    
    old_start_date = start_date - pd.DateOffset(days=365)
old_end_date = end_date - pd.DateOffset(days=365)
# Creating a Boolean index for the dataframe slice
    mask = (df[date_col] >= old_start_date) & (df[date_col] <= old_end_date)
    
    # Cutting out an additional piece of data
    # Using a Boolean index to get a slice
    result_df = df[mask].copy()

    # Adding to data_block_id
    if 'data_block_id' in df.columns:
        result_df['data_block_id'] = result_df['data_block_id'] + 365

    # Adding to the dates
    for col in add_date_cols:
        result_df[col] = result_df[col] + pd.DateOffset(days=365)
    
    df = pd.concat([df, result_df], ignore_index=True)
    df.to_csv(os.path.join(full_root_path, df_name), index=False0
```
- Adding an astronomical continuation of daylight hours in without a library =)
```
def calculate_solar_declination(day_of_year):
"""Calculation of solar declination."""
orbit_angle = (360 / 365.24) * (day_of_year - 81)
return 23.44 * math.sin(math.radians(orbit_angle))

    def calculate_hour_angle(latitude, solar_declination):
        """Calculation of the hour angle."""
latitude_rad = math.radians(latitude)
        solar_declination_rad = math.radians(solar_declination)
        cos_hour_angle = -math.tan(latitude_rad) * math.tan(solar_declination_rad)

        if cos_hour_angle < -1 or cos_hour_angle > 1:
return None # The sun does not rise or set
        return math.degrees(math.acos(cos_hour_angle))

    # Splitting the date string into year, month and day
    year, month, day = map(int, date_str.split('-'))
    date = datetime(year, month, day)
    day_of_year = date.timetuple().tm_yday

    solar_declination = calculate_solar_declination(day_of_year)
    hour_angle = calculate_hour_angle(latitude, solar_declination)

    if hour_angle is None:
        return 0 # Polar night or polar day
    day_length_hours = 2 * hour_angle / 15 # Day length in hours
    return day_length_hours * 60 # Duration of the day in minutes
```
- selection of blocks for models  (optuna parameters, the number of bins was limited, for faster models)
```
block = [[0,100,1],[100,100,0.5],[200,180,0.3],[380,1000,0.2]]
#block_cons1 = [[0,30,1],[30,70,0.7],[100,100,0.25],[200,180,0.15],[380,1000,0.1]]
block_cons1 = [[0,100,1],[100,100,0.5],[200,180,0.3],[380,1000,0.2]]
# block = [[0,10000,1]]


models_list = [
    {'model_name': 'model-1', 'new_model': lgb.LGBMRegressor(**p1), 'learn_again_period': 1, 'learn_again_offset': 0, 'is_consumption': 1, 'data_block_id_intervals': block_cons1, 'data_block_id_min': 0, 'drop_cols': cons1_drop_cols},
    {'model_name': 'model-2', 'new_model': lgb.LGBMRegressor(**p2), 'learn_again_period': 1, 'learn_again_offset': 0, 'is_consumption': 1, 'data_block_id_intervals': block_cons1, 'data_block_id_min': 0, 'drop_cols': cons1_drop_cols},
    {'model_name': 'model-3', 'new_model': lgb.LGBMRegressor(**p3), 'learn_again_period': 1, 'learn_again_offset': 0, 'is_consumption': 1, 'data_block_id_intervals': block_cons1, 'data_block_id_min': 0, 'drop_cols': cons1_drop_cols},

```

- this is to put it briefly.😀
