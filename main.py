import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
from networks.model import MrCAN
import time
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

filename = "SML2010.log"
logging.basicConfig(filename=filename, format='%(asctime)s %(filename)s %(levelname)s %(message)s',datefmt='%a %d %b %Y %H:%M:%S', filemode='w', level=logging.INFO)
raw_data = pd.read_csv("dataset/NEW-DATA-1.T15.txt", sep=' ')
target = '4:Temperature_Habitacion_Sensor'
cols = [
    '3:Temperature_Comedor_Sensor',
    '5:Weather_Temperature',
    '6:CO2_Comedor_Sensor',
    '7:CO2_Habitacion_Sensor',
    '8:Humedad_Comedor_Sensor',
    '9:Humedad_Habitacion_Sensor',
    '10:Lighting_Comedor_Sensor',
    '11:Lighting_Habitacion_Sensor',
    '12:Precipitacion',
    '13:Meteo_Exterior_Crepusculo',
    '14:Meteo_Exterior_Viento',
    '15:Meteo_Exterior_Sol_Oest',
    '16:Meteo_Exterior_Sol_Est',
    '17:Meteo_Exterior_Sol_Sud',
    '18:Meteo_Exterior_Piranometro',
    '22:Temperature_Exterior_Sensor',
    '23:Humedad_Exterior_Sensor',
    '24:Day_Of_Week']


depth = 64
batch_size = 256
prediction_horizon = 1
L = len(raw_data)
train_size = int(0.6*L)
val_size = int(0.2*L)
test_size = L - train_size - val_size

data_train = raw_data.iloc[:train_size]
data_val = raw_data.iloc[train_size:train_size+val_size]
data_test = raw_data.iloc[train_size+val_size:]

scaler_cols = MinMaxScaler(feature_range=(0, 1)).fit(data_train[cols].values)
data_train_cols_scale = scaler_cols.transform(data_train[cols].values)
data_val_cols_scale = scaler_cols.transform(data_val[cols].values)
data_test_cols_scale = scaler_cols.transform(data_test[cols].values)
data_train_cols_scale = pd.DataFrame(data_train_cols_scale)
data_val_cols_scale = pd.DataFrame(data_val_cols_scale)
data_test_cols_scale = pd.DataFrame(data_test_cols_scale)

scaler_target = MinMaxScaler(feature_range=(0, 1)).fit(np.expand_dims(data_train[target].values,axis=1))
data_train_target_scale = scaler_target.transform(np.expand_dims(data_train[target].values,axis=1))
data_val_target_scale = scaler_target.transform(np.expand_dims(data_val[target].values,axis=1))
data_test_target_scale = scaler_target.transform(np.expand_dims(data_test[target].values,axis=1))
data_train_target_scale = pd.DataFrame(data_train_target_scale)
data_val_target_scale = pd.DataFrame(data_val_target_scale)
data_test_target_scale = pd.DataFrame(data_test_target_scale)

# train
X1 = np.zeros((train_size, depth, len(cols)))
y_his1 = np.zeros((train_size, depth, 1))
y1 = np.zeros((train_size, 1))

for i, name in enumerate(data_train_cols_scale.columns):
    for j in range(depth):
        X1[:, j, i] = data_train_cols_scale[name].shift(depth - j - 1).fillna(method="bfill")
for j in range(depth):
    y_his1[:, j, :] = data_train_target_scale.shift(depth - j - 1).fillna(method="bfill")
y1 = data_train_target_scale.shift(- depth - prediction_horizon+1).fillna(method="bfill")

X_train = X1[depth-1:-prediction_horizon]
y_his_train = y_his1[depth-1:-prediction_horizon]
y_train = y1[:-depth-prediction_horizon+1]

del X1, y1, y_his1,data_train_cols_scale,data_train_target_scale
gc.collect()

# val
X2 = np.zeros((val_size, depth, len(cols)))
y_his2 = np.zeros((val_size, depth, 1))
y2 = np.zeros((val_size, 1))

for i, name in enumerate(data_val_cols_scale.columns):
    for j in range(depth):
        X2[:, j, i] = data_val_cols_scale[name].shift(depth - j - 1).fillna(method="bfill")
for j in range(depth):
    y_his2[:, j, :] = data_val_target_scale.shift(depth - j - 1).fillna(method="bfill")
y2 = data_val_target_scale.shift(- depth - prediction_horizon+1).fillna(method="bfill")

X_val = X2[depth-1:-prediction_horizon]
y_his_val = y_his2[depth-1:-prediction_horizon]
y_val = y2[:-depth-prediction_horizon+1]

del X2, y2, y_his2,data_val_cols_scale,data_val_target_scale
gc.collect()

# test
X3 = np.zeros((test_size, depth, len(cols)))
y_his3 = np.zeros((test_size, depth, 1))
y3 = np.zeros((test_size, 1))

for i, name in enumerate(data_test_cols_scale.columns):
    for j in range(depth):
        X3[:, j, i] = data_test_cols_scale[name].shift(depth - j - 1).fillna(method="bfill")
for j in range(depth):
    y_his3[:, j, :] = data_test_target_scale.shift(depth - j - 1).fillna(method="bfill")
y3 = data_test_target_scale.shift(- depth - prediction_horizon+1).fillna(method="bfill")

X_test = X3[depth-1:-prediction_horizon]
y_his_test = y_his3[depth-1:-prediction_horizon]
y_test = y3[:-depth-prediction_horizon+1]

del X3, y3, y_his3,data_test_cols_scale,data_test_target_scale
gc.collect()

X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
y_his_train_t = torch.Tensor(y_his_train)
y_his_val_t = torch.Tensor(y_his_val)
y_his_test_t = torch.Tensor(y_his_test)
y_train_t = torch.Tensor(y_train.values)
y_val_t = torch.Tensor(y_val.values)
y_test_t = torch.Tensor(y_test.values)
del X_train, X_val, X_test, y_his_train, y_his_val, y_his_test, y_train, y_val, y_test
gc.collect()

train_loader = DataLoader(TensorDataset(X_train_t, y_his_train_t, y_train_t), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(TensorDataset(X_val_t, y_his_val_t, y_val_t), shuffle=False, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_t, y_his_test_t, y_test_t), shuffle=False, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = MrCAN.add_model_specific_args()
args = parser.parse_args()
model = MrCAN(args).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)

epochs = 1000
loss = nn.MSELoss()
min_val_loss = 9999
train_rmse = []
logging.info("Window = "+str(depth))
logging.info("Horizon = "+str(prediction_horizon))
for i in range(epochs):
    mse_train = 0
    iteration_start = time.monotonic()
    model.train()
    for batch_x, batch_y_h, batch_y in train_loader:
        opt.zero_grad()
        y_pred1, y_pred2 = model(batch_x.to(device), batch_y_h.to(device))
        y_pred1 = y_pred1.squeeze(1)
        batch_y = batch_y.to(device).squeeze(1)
        if model.add_bf:
            y_pred2 = y_pred2.squeeze(1)
            l = (loss(y_pred1, batch_y) + loss(y_pred2, batch_y)) / 2.0
        else:
            l = loss(y_pred1, batch_y)
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    model.eval()
    with torch.no_grad():
        mse_val = 0
        for batch_x, batch_y_h, batch_y in val_loader:
            output, _ = model(batch_x.to(device), batch_y_h.to(device))
            output = output.squeeze(1)
            batch_y = batch_y.to(device).squeeze(1)
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]

    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        logging.info("Saving...")
        filename = "SML2010.pt"
        torch.save(model.state_dict(), filename)

    logging.info("Iter: " + str(i) + " train: " + str((mse_train / len(X_train_t)) ** 0.5) + " test: " + str(
        (mse_val / len(X_test_t)) ** 0.5))
    iteration_end = time.monotonic()
    logging.info("Iter time: " + str(iteration_end - iteration_start))
    train_rmse.append((mse_train / len(X_train_t)) ** 0.5)

# test
filename = "save/SML2010.pt"
model.load_state_dict(torch.load(filename))
model.eval()
with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    for batch_x, batch_y_h, batch_y in test_loader:
        output, _ = model(batch_x.to(device), batch_y_h.to(device))
        output = output.squeeze(1)
        batch_y = batch_y.to(device).squeeze(1)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        l = loss(output, batch_y).item()
        mse_val += l*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)

mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
mape = np.mean(np.abs((preds - true) / true)) * 100
t_mean = np.mean(true)
nse = 1 - np.sum((preds - true) ** 2) / np.sum((true - t_mean) ** 2)

logging.info('after normalization result:')
logging.info(mse**0.5)
logging.info(mae)
logging.info(mape)
logging.info(nse)

true = np.expand_dims(true,axis=1)
preds = np.expand_dims(preds,axis=1)
true = scaler_target.inverse_transform(true)
preds = scaler_target.inverse_transform(preds)
mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
mape = np.mean(np.abs((preds - true) / true)) * 100
t_mean = np.mean(true)
nse = 1 - np.sum((preds - true) ** 2) / np.sum((true - t_mean) ** 2)

logging.info('original data result:')
logging.info(mse**0.5)
logging.info(mae)
logging.info(mape)
logging.info(nse)
