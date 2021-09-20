import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_distinct_colors(models):
	color_map = plt.cm.get_cmap('viridis', len(models))
	color_list = [color_map(index) for index in range(len(models))]
	colors = dict(zip(models, color_list))
	return colors

df = pd.read_csv('all_losses.csv')
fig, _ = plt.subplots(nrows=2, ncols=2)
datasets = pd.unique(df['dataset'])
models = pd.unique(df['model'])
colors = get_distinct_colors(models)
legend = []
data_explanation = {'delaney': 'Solubility',
 'toxin': 'Toxicity',
 'cep': 'Photovoltaic Efficacy',
 'malaria': 'Drug Efficacy'}
model_explanation = {'conv_plus_net': 'Neural FPs + Neural Net',
 'conv_plus_linear': 'Neural FPs + single layer',
 'morgan_plus_net': 'Circular FPs + Neural Net',
 'morgan_plus_linear': 'Circular FPs + linear layer',
 'mean': 'Mean Predictor'}
for dataset, ax in zip(datasets, fig.axes):
	dataset_results = df.query('dataset == @dataset')
	for _, row in dataset_results.iterrows():
		ax.set_title(data_explanation[dataset])
		loss, model = row['loss'], row['model']
		line = ax.axhline(loss, color=colors[model])
		legend.append(tuple([line, model_explanation[model]]))
		# Remove x-ticks and x-labels
		ax.tick_params(axis='x', which='both', bottom=False,
		    top=False, labelbottom=False)
lines, model_explanations = zip(*legend)
fig.legend(lines, model_explanations[:len(models)])
plt.savefig('test_losses.png')
