import functools
import math
from PIL import ImageStat
import torchvision.transforms as transforms
import torch
import numpy as np

NUM_FEATURES_EXTRACTED = 4

# From non-normalized images
RED_STD = (2.5843577450538966, 98.80333849290223)
GREEN_STD = (2.2464895165870264, 90.61284418063957)
BLUE_STD = (2.409341710423115, 95.9119138864074)
HUE_STD = (0.0, 122.52893519024408)
SATURATION_STD = (0.0, 107.0988905957413)
INTENSITY_STD = (7.338606843665066, 95.46323981924607)
RED_VAR = (6.6789049544200605, 9762.099697343014)
GREEN_VAR = (5.046715148135411, 8210.687530504867)
BLUE_VAR = (5.804927477584582, 9199.09522535363)
HUE_VAR = (0.0, 15013.339958855035)
SATURATION_VAR = (0.0, 11470.172366838562)
INTENSITY_VAR = (53.85515040588774, 9113.230156786887)
ENTROPY = (1.6408668533310227, 10.585248012881095)
ENERGY = (0, 0.4827576579298601)
CONTRAST = (0, 3438.2474210218916)
HOMOGENEITY = (0.07557256790189121, 0.8689651476911414)

def extract_features(x, device):
    images = [transforms.ToPILImage()(e) for e in x]
    features = [get_image_features(i) for i in images]
    return torch.tensor(features, dtype=torch.float32, device=device)


# Image is PIL Image
def get_image_features(image):
    '''
    R, G, B = image.convert("RGB").split()
    H, S, _ = image.convert("HSV").split()
    I = image.convert("I")

    r_var = (ImageStat.Stat(R).var[0] - RED_VAR[0]) / (RED_VAR[1] - RED_VAR[0])
    r_std = (ImageStat.Stat(R).stddev[0] - RED_STD[0]) / (RED_STD[1] - RED_STD[0])

    g_var = (ImageStat.Stat(G).var[0] - GREEN_VAR[0]) / (GREEN_VAR[1] - GREEN_VAR[0])
    g_std = (ImageStat.Stat(G).stddev[0] - GREEN_STD[0]) / (GREEN_STD[1] - GREEN_STD[0])

    b_var = (ImageStat.Stat(B).var[0] - BLUE_VAR[0]) / (BLUE_VAR[1] - BLUE_VAR[0])
    b_std = (ImageStat.Stat(B).stddev[0] - BLUE_STD[0]) / (BLUE_STD[1] - BLUE_STD[0])

    i_var = (ImageStat.Stat(I).var[0] - INTENSITY_VAR[0]) / (INTENSITY_VAR[1] - INTENSITY_VAR[0])
    i_std = (ImageStat.Stat(I).stddev[0] - INTENSITY_STD[0]) / (INTENSITY_STD[1] - INTENSITY_STD[0])

    h_var = (ImageStat.Stat(H).var[0] - HUE_VAR[0]) / (HUE_VAR[1] - HUE_VAR[0])
    h_std = (ImageStat.Stat(H).var[0] - HUE_STD[0]) / (HUE_STD[1] - HUE_STD[0])

    s_var = (ImageStat.Stat(S).var[0] - SATURATION_VAR[0]) / (SATURATION_VAR[1] - SATURATION_VAR[0])
    s_std = (ImageStat.Stat(S).var[0] - SATURATION_STD[0]) / (SATURATION_STD[1] - SATURATION_STD[0])

    return [
        ImageStat.Stat(R).mean[0] / 256, r_var, r_std,
        ImageStat.Stat(G).mean[0] / 256, g_var, g_std,
        ImageStat.Stat(B).mean[0] / 256, b_var, b_std,
        ImageStat.Stat(H).mean[0] / 256, h_var, h_std,
        ImageStat.Stat(S).mean[0] / 256, s_var, s_std,
        ImageStat.Stat(I).mean[0] / 256, i_var, i_std,
    ]
    '''
    entropy, energy, contrast, homogeneity = get_more_features(np.array(image.convert("L")))

    entropy = (entropy - ENTROPY[0]) / (ENTROPY[1] - ENTROPY[0])
    energy = (energy - ENERGY[0]) / (ENERGY[1] - ENERGY[0])
    contrast = (contrast - CONTRAST[0]) / (CONTRAST[1] - CONTRAST[0])
    homogeneity = (homogeneity - HOMOGENEITY[0]) / (HOMOGENEITY[1] - HOMOGENEITY[0])

    return [
        #ImageStat.Stat(R).mean[0] / 256, r_var, r_std,
        #ImageStat.Stat(G).mean[0] / 256, g_var, g_std,
        #ImageStat.Stat(B).mean[0] / 256, b_var, b_std,
        #ImageStat.Stat(H).mean[0] / 256, h_var, h_std,
        #ImageStat.Stat(S).mean[0] / 256, s_var, s_std,
        #ImageStat.Stat(I).mean[0] / 256, i_var, i_std,
        entropy, energy, contrast, homogeneity
    ]



# image is grayscale_np
def get_more_features(image):
    IMAGE_SIZE = 224
    GRAY_LEVEL = 256
    displacement = [1, 0]

    glcm = [[0 for _ in range(GRAY_LEVEL)] for _ in range(GRAY_LEVEL)]

    row_max = IMAGE_SIZE - displacement[0] if displacement[0] else IMAGE_SIZE - 1
    col_max = IMAGE_SIZE - displacement[0] if displacement[0] else IMAGE_SIZE - 1

    for i in range(row_max):
        for j in range(col_max):
            m, n = image[i][j], image[i + displacement[0]][j + displacement[1]]
            glcm[m][n] += 1
            glcm[n][m] += 1

    normalizer = functools.reduce(lambda x, y: x + sum(y), glcm, 0)
    entropy, energy, contrast, homogeneity = 0, 0, 0, 0

    for m in range(len(glcm)):
        for n in range(len(glcm)):
            prob = (1.0 * glcm[m][n]) / normalizer
            log_prob = 0

            if 0.0001 <= prob <= 0.999:
                log_prob = math.log(prob, 2)

            entropy += -1.0 * prob * log_prob
            energy += prob ** 2
            contrast += ((m - n) ** 2) * prob
            homogeneity += prob / ((1 + abs(m - n)) * 1.0)

    if abs(entropy) < 0.0000001:
        entropy = 0

    return entropy, energy, contrast, homogeneity
