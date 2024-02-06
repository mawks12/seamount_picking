import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

gravity = pd.read_csv("/Users/seamount_picking/data/grav.xyz", sep="\t", names=["Longitude", "Latitude", "Intensity"])
test_gravity = gravity[gravity["Longitude"] >= -98]
test_gravity = test_gravity[test_gravity["Longitude"] <= -90]
test_gravity = test_gravity[test_gravity["Latitude"] >= -6]
test_gravity = test_gravity[test_gravity["Latitude"] <= -1.5]

test_gravity.to_csv('/Users/seamount_picking/data/test_grav.csv')