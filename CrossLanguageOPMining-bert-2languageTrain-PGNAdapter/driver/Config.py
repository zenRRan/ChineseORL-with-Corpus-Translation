from configparser import ConfigParser
import sys, os
sys.path.append('..')
#import models

class Configurable(object):
	def __init__(self, config_file, extra_args):
		config = ConfigParser()
		config.read(config_file)
		if extra_args:
			extra_args = dict([ (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
		for section in config.sections():
			for k, v in config.items(section):
				if k in extra_args:
					v = type(v)(extra_args[k])
					config.set(section, k, v)
		self._config = config
		if not os.path.isdir(self.save_dir):
			os.mkdir(self.save_dir)
		config.write(open(self.config_file,'w'))
		print('Loaded config file sucessfully.')
		for section in config.sections():
			for k, v in config.items(section):
				print(k, v)

	@property
	def pretrained_embeddings_file(self):
		return self._config.get('Data','pretrained_embeddings_file')
	@property
	def data_dir(self):
		return self._config.get('Data','data_dir')
	@property
	def source_train_file(self):
		return self._config.get('Data','source_train_file')
	@property
	def source_dev_file(self):
		return self._config.get('Data','source_dev_file')
	@property
	def source_test_file(self):
		return self._config.get('Data','source_test_file')
	@property
	def target_train_file(self):
		return self._config.get('Data', 'target_train_file')
	@property
	def target_dev_file(self):
		return self._config.get('Data', 'target_dev_file')
	@property
	def target_test_file(self):
		return self._config.get('Data', 'target_test_file')
	@property
	def min_occur_count(self):
		return self._config.getint('Data','min_occur_count')

	@property
	def bert_path(self):
		return self._config.get('bert', 'bert_path')

	@property
	def bert_config_path(self):
		return self._config.get('bert', 'bert_config_path')

	@property
	def bert_hidden_size(self):
		return self._config.getint('bert', 'bert_hidden_size')
	@property
	def output_attentions(self):
		return self._config.getboolean('bert', 'output_attentions')
	@property
	def output_hidden_states(self):
		return self._config.getboolean('bert', 'output_hidden_states')
	@property
	def tune_start_layer(self):
		return self._config.getint('bert', 'tune_start_layer')


	@property
	def use_adapter(self):
		return self._config.getboolean('AdapterPGN', 'use_adapter')

	@property
	def use_language_emb(self):
		return self._config.getboolean('AdapterPGN', 'use_language_emb')

	@property
	def num_adapters(self):
		return self._config.getint('AdapterPGN', 'num_adapters')

	@property
	def adapter_size(self):
		return self._config.getint('AdapterPGN', 'adapter_size')

	@property
	def one_hot(self):
		return self._config.getboolean('AdapterPGN', 'one_hot')

	@property
	def language_emb_size(self):
		return self._config.getint('AdapterPGN', 'language_emb_size')

	@property
	def language_emb_dropout(self):
		return self._config.getfloat('AdapterPGN', 'language_emb_dropout')

	@property
	def language_drop_rate(self):
		return self._config.getfloat('AdapterPGN', 'language_drop_rate')
	@property
	def num_language_features(self):
		return self._config.getint('AdapterPGN', 'num_language_features')

	@property
	def nl_project(self):
		return self._config.getint('AdapterPGN', 'nl_project')

	@property
	def language_features(self):
		return self._config.get('AdapterPGN', 'language_features')

	@property
	def in_langs(self):
		in_langs_list = self._config.get('AdapterPGN', 'in_langs').split(',')
		_in_langs_list = []
		for lang in in_langs_list:
			lang = lang.strip()
			_in_langs_list.append(lang)
		return _in_langs_list

	@property
	def out_langs(self):
		out_langs_list = self._config.get('AdapterPGN', 'out_langs').split(',')
		if out_langs_list[0] == '':
			return []
		_out_langs_list = []
		for lang in out_langs_list:
			lang = lang.strip()
			_out_langs_list.append(lang)
		return _out_langs_list

	@property
	def letter_codes(self):
		return self._config.get('AdapterPGN', 'letter_codes')








	@property
	def save_dir(self):
		return self._config.get('Save','save_dir')
	@property
	def config_file(self):
		return self._config.get('Save','config_file')
	@property
	def save_model_path(self):
		return self._config.get('Save','save_model_path')
	@property
	def save_vocab_path(self):
		return self._config.get('Save','save_vocab_path')
	@property
	def load_dir(self):
		return self._config.get('Save','load_dir')
	@property
	def load_model_path(self):
		return self._config.get('Save', 'load_model_path')
	@property
	def load_vocab_path(self):
		return self._config.get('Save', 'load_vocab_path')

	@property
	def model(self):
		return self._config.get('Network','model')
	@property
	def lstm_layers(self):
		return self._config.getint('Network','lstm_layers')
	@property
	def word_dims(self):
		return self._config.getint('Network','word_dims')
	@property
	def predict_dims(self):
		return self._config.getint('Network','predict_dims')
	@property
	def dropout_emb(self):
		return self._config.getfloat('Network','dropout_emb')
	@property
	def lstm_hiddens(self):
		return self._config.getint('Network','lstm_hiddens')
	@property
	def dropout_lstm_input(self):
		return self._config.getfloat('Network','dropout_lstm_input')
	@property
	def dropout_lstm_hidden(self):
		return self._config.getfloat('Network','dropout_lstm_hidden')
	@property
	def hidden_dims(self):
		return self._config.getint('Network','hidden_dims')
	@property
	def inner_hidden_dims(self):
		return self._config.getint('Network','inner_hidden_dims')
	@property
	def number_heads(self):
		return self._config.getint('Network','number_heads')
	@property
	def num_layers(self):
		return self._config.getint('Network','num_layers')
	@property
	def dropout_hidden(self):
		return self._config.getfloat('Network','dropout_hidden')
	
	@property
	def learning_rate(self):
		return self._config.getfloat('Optimizer','learning_rate')
	@property
	def decay(self):
		return self._config.getfloat('Optimizer','decay')
	@property
	def decay_steps(self):
		return self._config.getint('Optimizer','decay_steps')
	@property
	def beta_1(self):
		return self._config.getfloat('Optimizer','beta_1')
	@property
	def beta_2(self):
		return self._config.getfloat('Optimizer','beta_2')
	@property
	def epsilon(self):
		return self._config.getfloat('Optimizer','epsilon')
	@property
	def clip(self):
		return self._config.getfloat('Optimizer','clip')

	@property
	def num_buckets_train(self):
		return self._config.getint('Run','num_buckets_train')
	@property
	def num_buckets_valid(self):
		return self._config.getint('Run','num_buckets_valid')
	@property
	def num_buckets_test(self):
		return self._config.getint('Run','num_buckets_test')
	@property	
	def train_iters(self):
		return self._config.getint('Run','train_iters')
	@property	
	def train_batch_size(self):
		return self._config.getint('Run','train_batch_size')
	@property
	def test_batch_size(self):
		return self._config.getint('Run','test_batch_size')
	@property	
	def validate_every(self):
		return self._config.getint('Run','validate_every')
	@property
	def save_after(self):
		return self._config.getint('Run','save_after')
	@property
	def update_every(self):
		return self._config.getint('Run','update_every')

