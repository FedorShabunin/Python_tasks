import numpy as np
def gen(n: int, p: float):
    matrix = (np.random.random((n+2,n+2)) <= p).astype(int)
    matrix[0, :] *= 0
    matrix[n+1, :] *= 0
    matrix[:, 0] *= 0
    matrix[:, n+1] *= 0
    return matrix
    
p = .8
n = 10 

matrix = gen(n, p)


def movement_vectors(d):
    if d==2:
        directions = [np.array([1, 0]), np.array([0,1]), np.array([0,-1])]
        return (di for di in directions)
    
def search(M, pos):
    if M[pos[0], pos[1]] < 1:
        return False
    if pos[0] == M.shape[1] - 2 and M[pos[0], pos[1]]==1:
        return True
    M[pos[0], pos[1]] = -1
    ds = movement_vectors(2)
    return sum([search(M, pos + d) for d in ds]) > 0

def move_down(M):
    poses = [np.array([1,i]) for i in range(1, M.shape[0] - 1)]
    for pos in poses:
        if M[pos[0], pos[1]]:
            if search(M, pos):
                success = True
                return success
    return False
