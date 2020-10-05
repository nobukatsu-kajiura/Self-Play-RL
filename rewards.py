import numpy as np
import cv2

def BlackPadding(original, target):
    tmp = target[:, :]
    height, width_o = original.shape[:2]
    width_t = target.shape[1]

    start = int((width_o - width_t) / 2)
    fin = int((width_o + width_t) / 2)

    result = cv2.resize(np.zeros((1, 1, 3), np.uint8), (width_o, height))
    result[:height, start:fin] = tmp

    return result


def SquareBlackPadding(original, target=None):
    if target is not None:
        original = BlackPadding(target, original)
        
    tmp = original
    height, width = original.shape[:2]
    if(height > width and target is None):
        return 0

    start = int((width - height) / 2)
    fin = int((width + height) / 2)

    result = cv2.resize(np.zeros((1, 1, 3), np.uint8), (width, width))
    result[start:fin, :] = tmp

    return result


def difference(si, ti):
    return ((int(si[0]) - int(ti[0]))**2 + (int(si[1]) - int(ti[1]))**2 + (int(si[2]) - int(ti[2]))**2)/3


def AsymmetricDTW(s, t):
    M = np.zeros((s.shape[0]+1, t.shape[0]+1))
    for i in range(1, s.shape[0]+1):
        M[i,0] = np.inf
    for i in range(1, s.shape[0]+1):
        for j in range(1, t.shape[0]+1):
            M[i,j] = np.min([M[i-1, j-1] + difference(s[i-1], t[j-1]), M[i, j-1], M[i-1, j] + difference(s[i-1], t[j-1])])

    return M[len(s), len(t)]


def BDW(S, T):
    h = S.shape[0]
    ws = S.shape[1]
    wt = T.shape[1]
    StoT = 0
    TtoS = 0
    for i in range(h):
        StoT += AsymmetricDTW(S[i], T[i])
        TtoS += AsymmetricDTW(T[i], S[i])
    StoT /= h * ws
    TtoS /= h * wt

    return StoT + TtoS