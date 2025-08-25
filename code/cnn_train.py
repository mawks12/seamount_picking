from pathlib import Path
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import GeoDataset
from sklearn.neighbors import BallTree
import pandas as pd
from SeaNN import CNN, train, evaluate, SeamountDataset

torch.manual_seed(144)
device = torch.device('cpu')

data_path = Path('data/labled_pacific.nc')
data = xr.open_dataset(data_path)

model = CNN(kernel_size=9)

dataloader = DataLoader(SeamountDataset('data/mount_samples', 'data/vgg_swot.grd'), batch_size=1)
train_loss, train_acc = train(model, dataloader, 20, device)
avg_loss, accuracy = evaluate(model, dataloader=train_data, device=device)
torch.save(model, 'initial_model.pkl')
