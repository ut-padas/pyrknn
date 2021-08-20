
import numpy as np

def read_result(filename, m, k):
    with open('ref_avazu_full.txt') as f:
        flat_list=[word for line in f for word in line.split()]

    flat_list.reverse()
    print("Length: ", len(flat_list))

    nborID = np.zeros((m, k), dtype=np.int32)
    nborDist = np.zeros((m, k), dtype=np.float32)
    m = int(flat_list.pop())
    k = int(flat_list.pop())
    print(m, k)
    for i in range(m):
        for j in range(k):
            nborID[i, j] = int(flat_list.pop())


    for i in range(m):
        for j in range(k):
            nborDist[i, j] = float(flat_list.pop())
            
    return (nborID, nborDist)


result = read_result("ref_avazu_full.txt", 100, 64)

print(result)
