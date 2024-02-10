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

```python
# For the card on the T4x2 cable
!pip uninstall -y lightgbm
!pip install /kaggle/input/lightgbm420-cuda/lightgbm-4.2.0-py3-none-manylinux_2_35_x86_64.whl
import lightgbm as lgb

num_processes = 2
n_jobs = None
gpu_type = 'cuda'
gpus_n = 2
```
