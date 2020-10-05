import cv2
import numpy as np

def action_cr_l(original, pix):
    retargeted = original[:, :-pix]
    return retargeted

def action_cr_r(original, pix):
    retargeted = original[:, pix:]
    return retargeted

def action_scl(original, pix):
    retargeted = cv2.resize(original, dsize=(original.shape[1] - pix, original.shape[0]))
    return retargeted

def action_sc(original, pix):
    # The implementation of Seam Carving will not be published for licensing reasons.
    pass
    return original
