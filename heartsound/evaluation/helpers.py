import numpy as np


def calculate_mean_vector(emb_list):
    emb_list = np.array(emb_list)
    mean_vector = emb_list.mean(0)
    return mean_vector