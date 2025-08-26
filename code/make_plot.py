import matplotlib.pyplot as plt
import xarray as xr
import torch

device = torch.device('mps')
model = torch.load('initial_model.pkl', weights_only=False)
data_all = xr.open_dataset('data/mount_samples/sample_mh13.nc', engine='netcdf4')

data_pred = torch.as_tensor(data_all.z.values).reshape(
        1, 1, data_all.z.values.shape[0], data_all.z.values.shape[1]).to(device)

output = model(data_pred).cpu().detach().numpy().reshape(
        data_all.z.values.shape[0], data_all.z.values.shape[1])

data_all['outputs'] = xr.DataArray(output, dims=(
    data_all.dims['lon'], data_all.dims['lat']
    ))

def makeplot(data, name):
    plt.figure(figsize=(10, 10))
    flat_data = data.to_dataframe().reset_index()
    print(flat_data.head())
    plt.scatter(flat_data['lon'], flat_data['lat'], c=flat_data['z'])
    plt.title(name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    plt.savefig(f'{name}.png', dpi=300)


#makeplot(data_all.z, name='raw')

def makeplot(data, name):
    plt.figure(figsize=(10, 10))
    flat_data = data.to_dataframe().reset_index()
    print(flat_data.head())
    plt.scatter(flat_data[14400], flat_data[9600], c=flat_data['outputs'])
    plt.title(name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar()
    plt.savefig(f'{name}.png', dpi=300)
makeplot(data_all.outputs, name='output')
