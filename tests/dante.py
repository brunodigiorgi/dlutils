import dlutils
import numpy as np

vocabulary_size = 60
num_steps = 64
batch_size = 128
rnn_layers = [{'num_units': 32}]
nfolds = 5
max_epochs = 1
init_scale = 0.1
learning_rate = 0.1

with open('data/ladivinacommedia.txt', 'r') as f:
    data = f.read()
data = data.split(sep='\n\n')
data = [list(d) for d in data]

translator = dlutils.DatasetTranslator(data, vocabulary_size=vocabulary_size)
data_int = translator.to_int(data)
data_int = [np.array(d) for d in data_int]
nsymbols = len(translator.alphabet)

dense_layers = [{'num_units': nsymbols}]

enc = dlutils.OneHotEncoder(nsymbols)
dataset = dlutils.DatasetLanguageModel(data_int, num_steps, dataset_transformation=enc)

model = dlutils.RNNTensorFlow(num_steps, rnn_layers, dense_layers,
                              input_size=nsymbols, output_size=nsymbols,
                              init_scale=init_scale, learning_rate=learning_rate,
                              output_type='cls')

logger = dlutils.HtmlLogger("./log_dir", "loss")
test = dlutils.RNNTest(dataset, model, batch_size, logger, max_epochs=max_epochs)

test.test()
