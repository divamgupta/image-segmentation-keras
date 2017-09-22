


import h5py
import os


def loadWeightsPartial( model , weights_path , n_layers ):

	f = h5py.File(weights_path)
	for k in range(f.attrs['nb_layers']):
	    if k >= n_layers :
	        break
	    g = f['layer_{}'.format(k)]
	    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
	    model.layers[k].set_weights(weights)
	f.close()

	