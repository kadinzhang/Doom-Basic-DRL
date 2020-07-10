import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np


import random
import os
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt

# Functionality
from create_environment import *
from preprocess import *
from DQLearner import DQLearner
from Memory import Memory


game, possible_actions = create_environment()

# MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 500       # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob
# Q learning hyperparameters
gamma = 0.95               # Discounting rate
### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True



v1.reset_default_graph()
# Instantiate DQ Network, memory
DQLearner = DQLearner(state_size, action_size, learning_rate)
memory = Memory(max_size = memory_size)

# Pre-populate memory
game.new_episode()
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
state = None

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack(stacked_frames, state, True)

    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()

    if done:
        # Episode finished
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))

        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack(stacked_frames, next_state, False)

        memory.add((state, action, reward, next_state, done))
        state = next_state


### Train

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQLearner.loss)

write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQLearner.output, feed_dict={DQLearner.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


saver = v1.train.Saver()

if training == True:
    with v1.Session() as sess:
        sess.run(v1.global_variables_initializer())
        saver.restore(sess, "./models/model.ckpt")
        decay_step = 0
        game.init()

        for episode in range(total_episodes):
            step = 0

            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                reward = game.make_action(action)
                done = game.is_episode_finished()
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack(stacked_frames, next_state, False)
                    # End episode
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print(f"Episode: {episode}",
                          f"Total Reward: {total_reward}",
                          f"Training Loss: {loss:.4f}",
                          f"Explore Probability: {explore_probability:.4f}")
                    memory.add((state, action, reward, next_state, done))
                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                # Learn!

                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                Qs_next_state = sess.run(DQLearner.output, feed_dict = {DQLearner.inputs_: next_states_mb})

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                # just make it np
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQLearner.loss, DQLearner.optimizer],
                                   feed_dict={DQLearner.inputs_: states_mb,
                                              DQLearner.target_Q: targets_mb,
                                              DQLearner.actions_: actions_mb})
                summary = sess.run(write_op, feed_dict={DQLearner.inputs_: states_mb,
                                                        DQLearner.target_Q: targets_mb,
                                                        DQLearner.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


