import numpy as np
from skimage import transform
from collections import deque


def preprocess_frame(frame):
    # x = np.mean(frame, -1)
    # Crop roof, bottom, sides
    cropped = frame[30:-10, 30:-30]
    # Normalize
    normalized = cropped / 255.0
    # Resize to 84x84
    preprocessed_frame = transform.resize(normalized, [84, 84])
    return preprocessed_frame


stack_size = 4


def stack(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        for i in range(stack_size):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


