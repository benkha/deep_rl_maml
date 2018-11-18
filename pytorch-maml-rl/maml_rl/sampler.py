import gym
import torch
import multiprocessing as mp
import retrowrapper

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

# def make_env(env_name):
#     def _make_env():
#         return gym.make(env_name)
#     return _make_env

sonic_envs = {
    'SonicTheHedgehog-Genesis': [
        'GreenHillZone.Act1',
        'GreenHillZone.Act2',
        'GreenHillZone.Act3',
        'LabyrinthZone.Act1',
        'LabyrinthZone.Act2',
        'LabyrinthZone.Act3',
        'MarbleZone.Act1',
        'MarbleZone.Act2',
        'MarbleZone.Act3',
        'ScrapBrainZone.Act1',
        'ScrapBrainZone.Act2',
        'SpringYardZone.Act1',
        'SpringYardZone.Act2',
        'SpringYardZone.Act3',
        'StarLightZone.Act1',
        'StarLightZone.Act2',
        'StarLightZone.Act3',
    ],
    'SonicTheHedgehog2-Genesis': [
        'AquaticRuinZone.Act1',
        'AquaticRuinZone.Act2',
        'CasinoNightZone.Act1',
        'CasinoNightZone.Act2',
        'ChemicalPlantZone.Act1',
        'ChemicalPlantZone.Act2',
        'EmeraldHillZone.Act1',
        'EmeraldHillZone.Act2',
        'HilltopZone.Act1',
        'HilltopZone.Act2',
        'MetropolisZone.Act1',
        'MetropolisZone.Act2',
        'MysticCaveZone.Act1',
        'MysticCaveZone.Act2',
        'OilOceanZone.Act1',
        'OilOceanZone.Act2',
        'WingFortressZone',
    ],
    'SonicAndKnuckles3-Genesis': [
        'AngelIslandZone.Act1',
        'AngelIslandZone.Act2',
        'CarnivalNightZone.Act1',
        'CarnivalNightZone.Act2',
        'DeathEggZone.Act1',
        'DeathEggZone.Act2',
        'FlyingBatteryZone.Act1',
        'FlyingBatteryZone.Act2',
        'HiddenPalaceZone',
        'HydrocityZone.Act1',
        'HydrocityZone.Act2',
        'IcecapZone.Act1',
        'IcecapZone.Act2',
        'LaunchBaseZone.Act1',
        'LaunchBaseZone.Act2',
        'LavaReefZone.Act1',
        'LavaReefZone.Act2',
        'MarbleGardenZone.Act1',
        'MarbleGardenZone.Act2',
        'MushroomHillZone.Act1',
        'MushroomHillZone.Act2',
        'SandopolisZone.Act1',
        'SandopolisZone.Act2',
    ]
}

def make_env(env, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
