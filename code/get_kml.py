import sys
import pandas as pd

with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()

data = {
    'lat': [],
    'lon': [],
}
for line in lines:
    coords = line.strip().strip('<coordinates>').split(',')
    data['lat'].append(coords[1])
    data['lon'].append(coords[0])

data = pd.DataFrame(data)[['lat', 'lon']]
data.to_csv('extracted.csv', header=False, index=False)
