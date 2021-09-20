import json
import os
import pandas as pd
import numpy as np


neural_bounds = dict(fp_length      = [16, 128],   # Smaller upper range.
                     fp_depth       = [1, 4],
                     log_init_scale = [-2, -6],
                     log_step_size  = [-8, -4],
                     log_L2_reg     = [-6, 2],
                     h1_size        = [50, 100],
                     conv_width     = [5, 20])

datasets = ['delaney', 'toxin', 'malaria', 'cep']
hp_filenames = {dataset: '_'.join(['best_params', dataset, 'conv_plus_net'])\
                for dataset in datasets}
hp_dir = 'best_hyperparameters'
hp_files = {dataset: os.path.join(hp_dir, filename)\
            for dataset, filename in hp_filenames.items()}
params = []
lower_limits = []
upper_limits = []
for param, limits in neural_bounds.items():
	params.append(param)
	lower_limits.append(limits[0])
	upper_limits.append(limits[1])
df = pd.DataFrame({'lower_limit': lower_limits, 'upper_limit': upper_limits},
                  index=params)
for dataset in datasets:
	with open(hp_files[dataset]) as hp:
		results = json.load(hp)
		model = results['model']
		train = results['train']
		df.loc['fp_length', dataset] = model['fp_length']
		df.loc['fp_depth', dataset] = model['fp_depth']
		df.loc['log_init_scale', dataset] = np.log(train['init_scale'])
		df.loc['log_step_size', dataset] = np.log(train['step_size'])
		df.loc['log_L2_reg', dataset] = np.log(model['L2_reg'])
		df.loc['h1_size', dataset] = model['h1_size']
		df.loc['conv_width', dataset] = model['conv_width']
df.to_csv('hyperparameters.csv')
