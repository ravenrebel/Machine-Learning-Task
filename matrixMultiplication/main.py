import multiprocessing
import time

import numpy as np
import multiprocessing as mp


def dict(A, B, st, end, C):
    d1 = {}
    d1["A"] = A
    d1["B"] = B
    d1["_start"] = st
    d1["end"] = end
    d1["result"] = C
    return d1


def matrix_multiplication(d):
    # iterating by row of A
    for q in range(d["_start"], d["end"]):
        # iterating by col by B
        for p in range(len(d["B"][0])):
            # iterating by rows of B
            tmp_sum = 0
            for k in range(len(d["B"])):
                tmp_sum += d["A"][q][k] * d["B"][k][p]
            d["result"][q][p] += tmp_sum
    return d["result"]


if __name__ == '__main__':
    with multiprocessing.Pool(4) as pool:
        mx = np.random.rand(200, 200)
        my = np.random.rand(200, 200)
        # print(mx)
        # print(my)

        start = time.time()
        mc = []
        for i in range(len(mx)):
            row = []
            for j in range(len(my[0])):
                row.append(0)
            mc.append(row)

        d = {}
        d["A"] = mx
        d["B"] = my
        d["_start"] = 0
        d["end"] = len(mx)
        d["result"] = mc
        matrix_multiplication(d)

        stop = time.time()
        print(mc)
        print(stop - start)

        start = time.time()
        numpy_c = np.matmul(mx, my)
        stop = time.time()
        print(numpy_c)
        print(stop - start)

        start = time.time()
        mc = []
        for i in range(len(mx)):
            row = []
            for j in range(len(my[0])):
                row.append(0)
            mc.append(row)
        part1 = int(len(mx) / 4)
        part2 = int(len(mx) / 2)
        part3 = int(3 * len(mx) / 4)
        part4 = len(mx)
        d["_start"] = 0
        d["end"] = part1
        d["result"] = mc
        d1 = dict(mx, my, part1, part2, mc)
        d2 = dict(mx, my, part2, part3, mc)
        d3 = dict(mx, my, part3, part4, mc)

        p = pool.map(matrix_multiplication, [d, d1, d2, d3])
        print([p[0][0:part1], p[1][part1:part2]], p[2][part2:part3], p[3][part3:part4])
        stop = time.time()
        print(stop - start)

# /////////////////////////
