import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as v1
import tensorflow as tf
from collections import deque

# Functionality
from create_environment import *
from preprocess import *
from DQLearner import DQLearner


game, possible_actions = create_environment()

state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002

v1.reset_default_graph()
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
DQLearner = DQLearner(state_size, action_size, learning_rate)


with v1.Session() as sess:
    sess.run(v1.global_variables_initializer())
    totalScore = 0

    # Load the model
    game.init()
    for i in range(5):

        done = False

        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack(stacked_frames, state, True)

        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQLearner.output, feed_dict={DQLearner.inputs_: state.reshape((1, *state.shape))})
            saver = v1.train.Saver()
            saver.restore(sess, "./models/model.ckpt")

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()

            if done:
                break

            else:
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack(stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)
    game.close()