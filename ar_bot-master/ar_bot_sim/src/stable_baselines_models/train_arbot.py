import numpy as np
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

class TrainARBot:
    '''
    TrainARBot class is used to train the ARBot
    given an model, action space, and model
    '''

    def __init__(self, agent, env, actions, model, dense = False, action_mapping = None) -> None:
        self.agent = agent
        self.env = env
        self.actions = actions
        self.model = model
        self.action_mapping = action_mapping
        self.dense = dense

    def train(self, num_episodes, model_save_location = None, training_data_location = None, seed=43, obstacle = False) -> tuple:
            '''
            train function is used to train a model

            :param num_timesteps: how many timesteps to run
            :param obstacle: whether to spawn obstacles or not
            :param model_save_location: location to save the trained model, if none the model will not be saved
            :param training_data_location: where to save training data, if none data is not saved

            :return: a tuple consisting of two lists
            '''
            random_generator = np.random.default_rng(seed)

            env = self.env(self.agent, self.actions, self.action_mapping, random_generator, obstacle, self.dense)

            callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes)

            model = self.model("MlpPolicy", env, tensorboard_log="./ppo_arbot_tensorboard/", seed=seed)
            model.learn(int(1e10), callback=callback_max_episodes)
            
            env.close()

            total_sum_reward_tracker = env.total_sum_reward_tracker
            total_timestep_tracker = env.total_timestep_tracker

            del env

            # check if save location is specified, if so save last model for replaying
            if model_save_location is not None:
                model.save(model_save_location)

            # Convert to numpy array for saving
            total_sum_reward_tracker = np.array(total_sum_reward_tracker, dtype=np.float32)
            total_timestep_tracker = np.array(total_timestep_tracker, dtype=np.float32)

            # check if save location is specified, if so save training_data_location
            if training_data_location is not None:
                with open(training_data_location, 'wb') as training_data_file:
                    np.save(training_data_file, total_sum_reward_tracker)
                    np.save(training_data_file, total_timestep_tracker)

            return total_sum_reward_tracker, total_timestep_tracker
