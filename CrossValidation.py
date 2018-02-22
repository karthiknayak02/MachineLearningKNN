from math import *
import time
from pprint import pprint
import numpy as np
np.set_printoptions(suppress=True)
from tqdm import tqdm
from operator import itemgetter, attrgetter


def cosine_similarity(v1, v2):
    n = np.sum(v1 * v2)
    d1 = np.sqrt((np.sum(v1 * v1)))
    d2 = np.sqrt((np.sum(v2 * v2)))

    if n == 0 or d1 == 0 or d2 == 0:
        return 0

    return n / (d1 * d2)

def main():
    file = "data/u1-base.base"

    with open(file) as file:                                            # Extracting data
        data = [[int(x) for x in line.split()] for line in file]


    data = np.array(data)
    datatest = np.array(data)


    movies_n = int(max(data[:, 1]))          # 1682
    users_n = int(max(data[:, 0]))           # 943

    base_vectors = np.zeros((users_n, movies_n))

    for row in data:
        base_vectors[row[0]-1][row[1]-1] = float(row[2])

    k_values = [3, 4, 7, 8, 10]

    print("Determine the best K using cross validation")
    print("Using leave-one-out method")

    for i in range(5):

        k = k_values[i]
        print("\nChoosing K: " + str(k))
        ERRORs = []

        for rowt in tqdm(datatest):
            topR = []
            test_user_sims = []

            user_index = rowt[0] - 1
            movie_index = rowt[1] - 1

            rowb_index = 0
            for rowb in base_vectors:

                if rowb_index != (user_index) and rowb[movie_index] != 0:

                    sim = cosine_similarity(base_vectors[user_index], rowb)
                    test_user_sims.append([rowb_index, sim])

                rowb_index +=1

            top = sorted(test_user_sims, key=lambda x: (x[1]))[-k:]
            # top = sorted(test_user_sims, key=itemgetter(1))[-3:]

            for i in range(len(top)):
                topR.append(base_vectors[top[i][0]][movie_index])

            if len(top)!=0:
                top = np.array(top)
                Top3S = np.array(top[:, 1])
                Top3R = np.array(topR)

                prediction = np.sum(Top3S * Top3R) / np.sum(Top3S)
                ERRORs.append(np.square(prediction - rowt[2]))


        avg_error = np.average(ERRORs)
        print("\nAverage Error: " + str(avg_error))

main()