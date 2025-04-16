Train using the train_model.py file
Test the model using the test_model.py file

# Training Parameters (All optional):
    -n, --name NAME: Name the model. This is necessary if training multiple at once
                     Name of the model must end in .pth
    -c, --cont: Continue training from a previously trained model. REQUIRES --name
    -mt, --maxTime TIMESTEPS: Set the maximum timesteps to TIMESTEPS
    -me, --maxEp EPISODES: Set the maimum number of episodes to EPISODES
    -p, --pretrain: Pretrain the model on human trajectories. These trajectories
                    need to be in a file named trajectories.csv in this directory
    -r, --replay: Enable the replay buffer when training. Requires a trajectories.csv
                  file to be in this directory.

# Testing Parameters (All optional):
    -n, --name NAME: Name of the model. Only used if the model being tested was
                     named something custom
    -p, --path PATH: Path to the model. Only used if model was moved somewhere other
                     than default.
    -r, --redner: Enable rendering while testing
    --rand: Test the model against a opponent that takes random actions




# Commands for training on the server:

ssh capstone-2025@mulip-server.eecs.tufts.edu "cd ar_bot-master/ar_bot_sim/src/; python3 train_model.py > /dev/null 2>&1 < /dev/null &"

scp -P 22 -r capstone-2025@mulip-server.eecs.tufts.edu:/home/capstone-2025/ar_bot-master/ar_bot_sim/src/trained_models/ARBotGymEnv/* .