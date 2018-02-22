from math import *
import time
from pprint import pprint
import numpy as np
np.set_printoptions(suppress=True)
from tqdm import tqdm
from operator import itemgetter, attrgetter


def cosine_similarity(v1,v2):
    # print(v1, v2)

    n = np.sum(v1 * v2)
    d1 = sqrt(np.sum(v1 * v1))
    d2 = sqrt(np.sum(v2 * v2))

    if n == 0 or d1 == 0 or d2 == 0:
        return 0

    return n / (d1 * d2)


def main():
    start = time.time()
    file = "data/u1-base.base"

    with open(file) as file:
        data = [[int(x) for x in line.split()] for line in file]


    data = np.array(data)


    movies_n = int(max(data[:, 1]))          # 1682
    users_n = int(max(data[:, 0]))           # 943


    base_vectors = np.zeros((users_n, movies_n))

    for row in data:
        base_vectors[row[0]-1][row[1]-1] = float(row[2])

    # print(base_vectors)
    # print()

    #NORMAILZATION_______________UNUSED___________________________________________________
    #base_vectors = np.zeros((users_n, movies_n)).astype('object')
    #
    #for row in data:
    #    base_vectors[row[0]-1][row[1]-1] = Fraction(row[2])
    #    
    #    print(base_vectors)
    #    print()
    #    
    #    for i in range(users_n):
    #        num = int(np.sum(base_vectors[i]))
    #        den = movies_n - base_vectors[i].tolist().count(0)
    #        for j in range(movies_n):
    #            if base_vectors[i][j] != 0:
    #                base_vectors[i][j] -= Fraction(num, den)
    #
    #print(base_vectors)
    #print()

    #NORMAILZATION________________________________________________________________________


    #___________________TEST______________
    filetest = "data/u1-test.test"

    with open(filetest) as filetest:
        datatest = [[int(x) for x in line.split()] for line in filetest]

    datatest = np.array(datatest)
    # print(datatest)

    test_ratings = datatest[:, 2]
    mainR = []

    print("User-Based Collaborative Filtering")
    print("Choosing K: 3")

    for rowt in tqdm(datatest):
        topR = []
        test_user_sims = []
        rowb_index = 0
        for rowb in base_vectors:

            if rowb[rowt[1]-1] != 0 and rowb_index != rowt[0] - 1:

                sim = cosine_similarity(base_vectors[rowt[0]-1], rowb)
                user = rowb_index + 1

                test_user_sims.append([user, sim])


            rowb_index +=1


        # print(test_user_sims)
        # top = sorted(test_user_sims, key=lambda x: (x[1]))[-3:]
        top = sorted(test_user_sims, key=itemgetter(1))[-3:]
        # print(top)

        for i in range(len(top)):
            topR.append(base_vectors[top[i][0]-1][rowt[1]-1])
        # print(topR)

        if len(top)!=0:
            top = np.array(top)
            Top3S = np.array(top[:, 1])
            # print(Top3S)
            Top3R = np.array(topR)
            # print(Top3R)

            mainR.append(np.sum(Top3S * Top3R) / np.sum(Top3S))
        else:
            mainR.append(2.5)

    avg_error = np.sum(np.square(mainR - test_ratings)) / len(datatest)
    print("\nAverage mean squared error: " + str(avg_error))

main()
