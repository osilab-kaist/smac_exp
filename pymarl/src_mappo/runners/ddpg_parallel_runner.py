# from envs import REGISTRY as env_REGISTRY
# from functools import partial
# from components.episode_buffer import EpisodeBatch
# from multiprocessing import Pipe, Process
# import numpy as np
# import torch as th


# # Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# # https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# class DDPG_ParallelRunner:

#     def __init__(self, args, logger):
#         self.args = args
#         self.logger = logger
#         self.batch_size = self.args.batch_size_run

#         # Make subprocesses for the envs
#         self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
#         env_fn = env_REGISTRY[self.args.env]
#         self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
#                             for worker_conn in self.worker_conns]

#         for p in self.ps:
#             p.daemon = True
#             p.start()

#         self.parent_conns[0].send(("get_env_info", None))
#         self.env_info = self.parent_conns[0].recv()
#         self.episode_limit = self.env_info["episode_limit"]

#         self.t = 0

#         self.t_env = 0

#         self.train_returns = []
#         self.test_returns = []
#         self.train_stats = {}
#         self.test_stats = {}

#         self.log_train_stats_t = -100000

#     def setup(self, scheme, groups, preprocess, mac):
#         self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
#                                  preprocess=preprocess, device=self.args.device)
#         self.mac = mac
#         self.scheme = scheme
#         self.groups = groups
#         self.preprocess = preprocess

#     def get_env_info(self):
#         return self.env_info

#     def save_replay(self):
#         pass

#     def close_env(self):
#         for parent_conn in self.parent_conns:
#             parent_conn.send(("close", None))

#     def reset(self):
#         self.batch = self.new_batch()

#         # Reset the envs
#         for parent_conn in self.parent_conns:
#             parent_conn.send(("reset", None))

#         pre_transition_data = {
#             "state": [],
#             "avail_actions": [],
#             "obs": []
#         }
#         # Get the obs, state and avail_actions back
#         for parent_conn in self.parent_conns:
#             data = parent_conn.recv()
#             pre_transition_data["state"].append(data["state"])
#             pre_transition_data["avail_actions"].append(data["avail_actions"])
#             pre_transition_data["obs"].append(data["obs"])

#         self.batch.update(pre_transition_data, ts=0)

#         self.t = 0
#         self.env_steps_this_run = 0

#     def run(self, test_mode=False):
#         self.reset()

#         all_terminated = False
#         episode_returns = [0 for _ in range(self.batch_size)]
#         episode_lengths = [0 for _ in range(self.batch_size)]
#         self.mac.init_hidden(batch_size=self.batch_size)
#         terminated = [False for _ in range(self.batch_size)]
#         envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
#         final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

#         while True:

#             # Pass the entire batch of experiences up till now to the agents
#             # Receive the actions for each agent at this timestep in a batch for each un-terminated env
#             actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
#             cpu_actions = actions.to("cpu").detach().numpy()

#             # Update the actions taken
#             actions_chosen = {
#                 "actions": actions.unsqueeze(1)
#             }
#             print('action unsqueeze::::', actions.unsqueeze(1).shape)
#             self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

#             # Send actions to each env
#             action_idx = 0
#             for idx, parent_conn in enumerate(self.parent_conns):
#                 if idx in envs_not_terminated: # We produced actions for this env
#                     if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
#                         parent_conn.send(("step", cpu_actions[action_idx]))
#                     action_idx += 1 # actions is not a list over every env

#             # Update envs_not_terminated
#             envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
#             all_terminated = all(terminated)
#             if all_terminated:
#                 break

#             # Post step data we will insert for the current timestep
#             post_transition_data = {
#                 "reward": [],
#                 "terminated": []
#             }
#             # Data for the next step we will insert in order to select an action
#             pre_transition_data = {
#                 "state": [],
#                 "avail_actions": [],
#                 "obs": []
#             }

#             # Receive data back for each unterminated env
#             for idx, parent_conn in enumerate(self.parent_conns):
#                 if not terminated[idx]:
#                     data = parent_conn.recv()
#                     # Remaining data for this current timestep
#                     post_transition_data["reward"].append((data["reward"],))

#                     episode_returns[idx] += data["reward"]
#                     episode_lengths[idx] += 1
#                     if not test_mode:
#                         self.env_steps_this_run += 1

#                     env_terminated = False
#                     if data["terminated"]:
#                         final_env_infos.append(data["info"])
#                     if data["terminated"] and not data["info"].get("episode_limit", False):
#                         env_terminated = True
#                     terminated[idx] = data["terminated"]
#                     post_transition_data["terminated"].append((env_terminated,))

#                     # Data for the next timestep needed to select an action
#                     pre_transition_data["state"].append(data["state"])
#                     pre_transition_data["avail_actions"].append(data["avail_actions"])
#                     pre_transition_data["obs"].append(data["obs"])

#             # Add post_transiton data into the batch
#             self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

#             # Move onto the next timestep
#             self.t += 1

#             # Add the pre-transition data
#             self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

#         if not test_mode:
#             self.t_env += self.env_steps_this_run

#         # Get stats back for each env
#         for parent_conn in self.parent_conns:
#             parent_conn.send(("get_stats",None))

#         env_stats = []
#         for parent_conn in self.parent_conns:
#             env_stat = parent_conn.recv()
#             env_stats.append(env_stat)

#         cur_stats = self.test_stats if test_mode else self.train_stats
#         cur_returns = self.test_returns if test_mode else self.train_returns
#         log_prefix = "test_" if test_mode else ""
#         infos = [cur_stats] + final_env_infos
#         cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
#         cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
#         cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

#         cur_returns.extend(episode_returns)

#         n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
#         if test_mode and (len(self.test_returns) == n_test_runs):
#             self._log(cur_returns, cur_stats, log_prefix)
#         elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
#             self._log(cur_returns, cur_stats, log_prefix)
#             if hasattr(self.mac.action_selector, "epsilon"):
#                 self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
#             self.log_train_stats_t = self.t_env

#         return self.batch

#     def _log(self, returns, stats, prefix):
#         self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
#         self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
#         returns.clear()

#         for k, v in stats.items():
#             if k != "n_episodes":
#                 self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
#         stats.clear()


# def env_worker(remote, env_fn):
#     # Make environment
#     env = env_fn.x()
#     while True:
#         cmd, data = remote.recv()
#         if cmd == "step":
#             actions = data
#             # Take a step in the environment
#             reward, terminated, env_info = env.step(actions)
#             # Return the observations, avail_actions and state to make the next action
#             state = env.get_state()
#             avail_actions = env.get_avail_actions()
#             obs = env.get_obs()
#             remote.send({
#                 # Data for the next timestep needed to pick an action
#                 "state": state,
#                 "avail_actions": avail_actions,
#                 "obs": obs,
#                 # Rest of the data for the current timestep
#                 "reward": reward,
#                 "terminated": terminated,
#                 "info": env_info
#             })
#         elif cmd == "reset":
#             env.reset()
#             remote.send({
#                 "state": env.get_state(),
#                 "avail_actions": env.get_avail_actions(),
#                 "obs": env.get_obs()
#             })
#         elif cmd == "close":
#             env.close()
#             remote.close()
#             break
#         elif cmd == "get_env_info":
#             remote.send(env.get_env_info())
#         elif cmd == "get_stats":
#             remote.send(env.get_stats())
#         else:
#             raise NotImplementedError


# class CloudpickleWrapper():
#     """
#     Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
#     """
#     def __init__(self, x):
#         self.x = x
#     def __getstate__(self):
#         import cloudpickle
#         return cloudpickle.dumps(self.x)
#     def __setstate__(self, ob):
#         import pickle
#         self.x = pickle.loads(ob)








############################ ddpg_parallel_runner#####################################
from gym import spaces
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.ddpg_episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import time


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class DDPG_ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        if self.args.env == "sc2":
            self.ps = [Process(target=env_worker,
                               args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                       for worker_conn in self.worker_conns]
        else:
            self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, env_args=self.args.env_args, args=self.args))))
                                for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        # self.train_returns_total = []
        # self.test_returns_total = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        self.last_learn_T = 0
        pass

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch,
                                 scheme,
                                 groups,
                                 self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, **kwargs):
        self.reset()

        if not test_mode:
            parent_conns = self.parent_conns #self.parent_conns[0:1]
        else:
            parent_conns = self.parent_conns

        all_terminated = False
        if not test_mode:
            n_parallel_envs = self.batch_size
        else:
            n_parallel_envs = self.batch_size
        episode_returns = [0 for _ in range(n_parallel_envs)]
        episode_lengths = [0 for _ in range(n_parallel_envs)]
        self.mac.init_hidden(batch_size=n_parallel_envs)
        terminated = [False for _ in range(n_parallel_envs)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        action_norms = []
        action_means = []
        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").detach().numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Post step data we will insert for the current timestep
            post_transition_data = {
                # "actions": actions.unsqueeze(1),
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Update terminated envs after adding post_transition_data
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data

            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            # learner = kwargs.get("learner")
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
            #                                   test_mode=test_mode)
            # cpu_actions = actions.to("cpu").detach().numpy()
            # action_norms.append(np.sqrt(np.sum(cpu_actions**2)))
            # action_means.append(np.mean(cpu_actions))

            # # Update the actions taken
            # actions_chosen = {
            #     "actions": actions.unsqueeze(1)
            # }

            # self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # # Send actions to each env
            # action_idx = 0

            # for idx, parent_conn in enumerate(parent_conns):
            #     if idx in envs_not_terminated: # We produced actions for this env
            #         if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
            #             parent_conn.send(("step", cpu_actions[action_idx]))
            #         action_idx += 1 # actions is not a list over every env

            # # Post step data we will insert for the current timestep
            # post_transition_data = {
            #     # "actions": actions.unsqueeze(1),
            #     "reward": [],
            #     "terminated": []
            # }
            # # Data for the next step we will insert in order to select an action
            # pre_transition_data = {
            #     "state": [],
            #     "avail_actions": [],
            #     "obs": []
            # }

            # # Update terminated envs after adding post_transition_data
            # envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            # all_terminated = all(terminated)
            # if all_terminated:
            #     break

            # # Receive data back for each unterminated env
            # for idx, parent_conn in enumerate(parent_conns):
            #     if not terminated[idx]:
            #         data = parent_conn.recv()
            #         # Remaining data for this current timestep
            #         post_transition_data["reward"].append((data["reward"],))

            #         # only cooperative scenarios!
            #         episode_returns[idx] += data["reward"]
            #         episode_lengths[idx] += 1
            #         if not test_mode:
            #             self.env_steps_this_run += 1

            #         env_terminated = False
            #         if data["terminated"]:
            #             final_env_infos.append(data["info"])
            #         if data["terminated"] and not data["info"].get("episode_limit", False):
            #             env_terminated = True
            #         terminated[idx] = data["terminated"]
            #         post_transition_data["terminated"].append((env_terminated,))

            #         # Data for the next timestep needed to select an action
            #         pre_transition_data["state"].append(data["state"])
            #         pre_transition_data["avail_actions"].append(data["avail_actions"])
            #         pre_transition_data["obs"].append(data["obs"])

            # # Add post_transiton data into the batch
            # self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # if self.args.verbose:
            #     print("Transition nr {} in episode {} now filled in...".format(self.t, kwargs.get("episode", "?")))

            # # Move onto the next timestep
            # self.t += 1

            # self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

            if (not test_mode) and (getattr(self.args, "runner_scope", "episodic") == "transition"):
                buffer = kwargs.get("buffer")
                learner = kwargs.get("learner")
                episode = kwargs.get("episode")

                # insert single transitions into buffer
                # note zeros inserted for batch elements already terminated
                # buffer.insert_episode_batch(self.batch[:, self.t-1:self.t+1])
                buffer.insert_episode_batch(self.batch[0, self.t - 1:self.t + 1])

                if (self.t_env + self.t - self.last_learn_T) / self.args.learn_interval >= 1.0:
                    # execute learning steps (if enabled)
                    if buffer.can_sample(self.args.batch_size) and (buffer.episodes_in_buffer > getattr(self.args, "buffer_warmup", 0)):
                        episode_sample = buffer.sample(self.args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != self.args.device:
                            episode_sample.to(self.args.device)

                        if self.args.verbose:
                            print("Learning now for {} steps...".format(getattr(self.args, "n_train", 1)))
                        for _ in range(getattr(self.args, "n_train", 1)):
                            learner.train(episode_sample, self.t_env, episode)
                        self.last_learn_T = self.t_env + self.t

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})

        cur_stats["n_episodes"] = n_parallel_envs + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        cur_stats["action_norms"] = np.mean(action_norms) + cur_stats.get("action_norms", 0)
        cur_stats["action_means"] = np.mean(action_means) + cur_stats.get("action_means", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif (not test_mode) and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac, "action_selector") and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        self.mac.ou_noise_state = actions.clone().zero_()

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            data = np.argmax(data, axis = 1)
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            if isinstance(reward, (list, tuple)):
                assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                reward = reward[0]
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        # elif cmd == "agg_stats":
        #     agg_stats = env.get_agg_stats(data)
        #     remote.send(agg_stats)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)