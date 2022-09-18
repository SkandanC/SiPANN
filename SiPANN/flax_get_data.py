from time import time
import numpy as np
from SiPANN.nn import evWGcoupler

wavelength = np.linspace(1.45, 1.65, 20)
width      = np.linspace(0.4, 0.6, 2)
thickness  = np.linspace(0.18, 0.24, 2)
sw_angle   = np.linspace(80, 90, 2)
gap        = np.linspace(0.05,0.3,2)
derivative = 1
start = time()
TE0,TE1 = evWGcoupler(wavelength,width,thickness,gap,sw_angle)
end = time()
print(end - start)
print(TE0.shape, TE1.shape)
np.savez("SiPANN\\FLAX_DATA\\COUPLER_GAP\\data_dummy.npz", TE0=TE0, TE1=TE1)
