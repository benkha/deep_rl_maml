"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""
import pickle

import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import tensorflow as tf
import baselines.ppo2.ppo2 as ppo2
import gym_remote.exceptions as gre

from sonic_util import make_env, sample_env

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    env = sample_env()

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        # env = DummyVecEnv([make_env])

        model, all_returns = ppo2.learn(network='cnn',
                   env=env,
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(3e5),
                   save_interval=10)

        pickle.dump(all_returns, open("all_returns.pkl", "wb"))
                   # load_path='/var/folders/sl/wj836d4x7f11jncxpz3n2c300000gn/T/openai-2018-10-31-16-44-55-400600/checkpoints/00109')

        # runner = ppo2.Runner(env=env, model=model, nsteps=2048, gamma=0.99, lam=0.95)
        # eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = runner.run()
        # print("Runner returns", eval_returns)
        # print("Mean returns", np.mean(eval_returns))

if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except gre.GymRemoteError as exc:
    #     print('exception', exc)
