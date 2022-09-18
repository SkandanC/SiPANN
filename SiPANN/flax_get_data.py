import numpy as np
import jax.numpy as numpy
from SiPANN.nn import evWGcoupler

wavelength = np.linspace(1.45, 1.65, 20)
width      = np.linspace(0.4, 0.6, 10)
thickness  = np.linspace(0.18, 0.24, 10)
sw_angle   = np.linspace(80, 90, 10)
gap        = np.linspace(0.05,0.3,10)
derivative = 1
TE0,TE1 = evWGcoupler(wavelength,width,thickness,gap,sw_angle)
print(TE0.shape, TE1.shape)
numpy.savez("SiPANN\\FLAX_DATA\\COUPLER_GAP\\data.npz", TE0=TE0, TE1=TE1)
