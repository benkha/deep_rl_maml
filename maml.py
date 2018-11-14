import tensorflow as tf
import numpy as np

import time
from sonic_util import make_env, sample_env

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

class MAML(object):
	def __init__(self, computation_graph_args):
		self.alpha = computation_graph_args['alpha']
		self.beta = computation_graph_args['beta']
		self.ob_dim = computation_graph_args['ob_dim']
		self.ac_dim = computation_graph_args['ac_dim']
		self.batch_size = computation_graph_args['batch_size']
		self.horizon = computation_graph_args['horizon']
		self.n_trajectories = computation_graph_args['n_trajectories']
        self.is_train = computation_graph_args['is_train']
        self.model_type = computation_graph_args['model_type']
        self.loss_type = computation_graph_args['loss_type']

        self.meta_train_obs, self.meta_train_ac, self.meta_val_obs, self.meta_val_ac \
            = self.define_placeholders()

        self.meta_optimizer = tf.train.AdamOptimizer(self.beta)
        self.model = self.import_model()
        self.loss_fn = self.import_loss_fn()
        self.build_computation_graph()

	def init_tf_sess(self):
		tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def define_placeholders(self):
    	meta_train_obs = tf.placeholder(shape=[None, self.ob_dim], name="train_ob", dtype=tf.float32)
        meta_train_ac = tf.placeholder(shape=[None, self.ac_dim], name="train_ac", dtype=tf.float32)
        meta_val_obs = tf.placeholder(shape=[None, self.ob_dim], name="val_ob", dtype=tf.float32)
        meta_val_ac = tf.placeholder(shape=[None, self.ac_dim], name="val_ac", dtype=tf.float32)

    def build_computation_graph(self):
        self.loss = 0.0 # TODO: build full computation graph, I think this should depend on using a MLP to sample actions not sure
    	self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

    def import_model(self):
        if self.model_type == 'fc':
            import model.fc as model
        else:
            ValueError("Can't recognize the model type {}".format(self.model_type))
        return model

    def import_loss_fn(self):
        if self.loss_type == 'mse':
            self.loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(self.loss_type))

    def metalearn(self):
        pass

    def evaluate(self):
        pass

    def train_step(self):
        pass

    def test_step(self):
        pass

def train_MAML(
		env_name,
		exp_name,
        model_type,
        loss_type,
		render,
		n_iter
		alpha,
		beta,
		batch_size,
		horizon,
		n_trajectories,
        seed,
        is_train
		):
	# Initialize environment
    np.random.seed(seed)
	env = sample_env()
    env.reset()
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

	computation_graph_args = {
        'alpha': alpha,
        'beta': beta,
        'ob_dim': ob_dim
        'ac_dim': ac_dim
        'batch_size': batch_size
        'horizon': horizon
        'n_trajectories': n_trajectories
        'is_train': is_train
        'model_type': model_type
        'loss_type': loss_type
    }

    # Initialize metalearner
    maml_agent = MAML(computation_graph_args)
    maml_agent.init_tf_sess()
    if is_train:
        maml_agent.metalearn() # TODO: insert max steps?
    else:
        maml_agent.evaluate() # TODO: params


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--model_type', type=str, default='fc')
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--horizon', '-h', type=int, default=100)
    parser.add_argument('--n_trajectories', '-t', type=int, default=10)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--is_train', action='store_true')
	args = parser.parse_args()

	data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'maml_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

	train_MAML(
		env_name=args.env_name,
		exp_name=args.exp_name,
        model_type=args.model_type,
        loss_type=args.loss_type,
		render=args.render,
		n_iter=args.n_iter,
		alpha=args.alpha,
		beta=args.beta,
		batch_size=args.batch_size,
		logdir=logdir,
		horizon=horizon,
		n_trajectories=n_trajectories,
        seed=args.seed,
        is_train=args.is_train
	)

if __name__ == '__main__':
    main()
