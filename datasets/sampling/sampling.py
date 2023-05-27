import numpy as np
from rtm_torch.rtm import RTM
para_dict = {}
para_names = ["N", "cab", "cw", "cm", "LAI", "typeLIDF", "LIDF",
              "hspot", "psoil", "tts", "tto", "psi", "cp", "cbc",
              "car", "anth", "cbrown", "LAIu", "cd", "sd", "h"]
para_learnable = [""]