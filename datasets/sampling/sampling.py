import os
import numpy as np
import torch
from rtm_torch.rtm import RTM
import pandas as pd
torch.manual_seed(0)

SAVE_PATH = "/maps/ys611/ai-refined-rtm/data/synthetic/20230529"
# B01 and B10 will not be used in the training
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                 'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                 'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                 'B12_SWI2']


def uniform_sampling(min, max, num_samples):
    # uniform sampling using pytorch
    return torch.rand(num_samples) * (max - min) + min


def para_sampling(num_samples=100):
    # run uniform sampling for learnable parameters
    para_dict = {}
    # # Brightness Factor (psoil) when using default soil spectrum
    # para_dict["psoil"] = np.random.uniform(0.0, 1.0, num_samples)

    # Leaf Model Parameters
    # N: Structure Parameter (N)
    para_dict["N"] = uniform_sampling(1.0, 4.0, num_samples)
    # cab: Chlorophyll A+B (cab)
    para_dict["cab"] = uniform_sampling(0.0, 100.0, num_samples)
    # cw: Water Content (Cw)
    para_dict["cw"] = uniform_sampling(0.002, 0.08, num_samples)
    # cm: Dry Matter (cm)
    para_dict["cm"] = uniform_sampling(0.0, 0.05, num_samples)
    # # car: Carotenoids (Ccx)
    # para_dict["car"] = np.random.uniform(0.0, 30.0, num_samples)
    # # cbrown: Brown Pigments (Cbrown)
    # para_dict["cbrown"] = np.random.uniform(0.0, 1.0, num_samples)
    # # anth: Anthocyanins (Canth)
    # para_dict["anth"] = np.random.uniform(0, 5, num_samples)
    # # cp: Proteins (Cp)
    # para_dict["cp"] = np.random.uniform(0.0, 0.01, num_samples)
    # # cbc: Carbon-based constituents (CBC)
    # para_dict["cbc"] = np.random.uniform(0.0, 0.01, num_samples)

    # Canopy Model Parameters
    # LAI: (Single) Leaf Area Index (LAI)
    para_dict["LAI"] = uniform_sampling(0.01, 15.0, num_samples)
    # # typeLIDF: Leaf Angle Distribution (LIDF) type: 1 = Beta, 2 = Ellipsoidal
    # # if typeLIDF = 2, LIDF is set to between 0 and 90 as Leaf Angle to calculate the Ellipsoidal distribution
    # # if typeLIDF = 1, LIDF is set between 0 and 5 as index of one of the six Beta distributions
    # para_dict["typeLIDF"] = np.full(num_samples, 2)
    # LIDF: Leaf Angle (LIDF), only used when LIDF is Ellipsoidal
    para_dict["LIDF"] = uniform_sampling(0.0, 90.0, num_samples)
    # # hspot: Hot Spot Size Parameter (Hspot)
    # para_dict["hspot"] = np.random.uniform(0.0, 1.0, num_samples)
    # # tto: Observation zenith angle (Tto)
    # para_dict["tto"] = np.random.uniform(0.0, 89.0, num_samples)
    # # tts: Sun zenith angle (Tts)
    # para_dict["tts"] = np.random.uniform(0.0, 89.0, num_samples)
    # # psi: Relative azimuth angle (Psi)
    # para_dict["psi"] = np.random.uniform(0.0, 180.0, num_samples)

    # Forest Model Parameters
    # LAIu: Undergrowth LAI (LAIu)
    para_dict["LAIu"] = uniform_sampling(0.01, 3.0, num_samples)
    # sd: Stem Density (SD)
    para_dict["sd"] = uniform_sampling(0.0, 3000.0, num_samples)
    # h: Tree Height (H)
    para_dict["h"] = uniform_sampling(1.0, 50.0, num_samples)
    # cd: Crown Diameter (CD)
    para_dict["cd"] = uniform_sampling(1.0, 15.0, num_samples)

    return para_dict


def run_sampling():
    # sample the dataset and save it to a csv file
    rtm = RTM()
    for i in range(180):
        para_dict = para_sampling(num_samples=100)
        # run the RTM without tracking gradients
        with torch.no_grad():
            spectra = rtm.run(**para_dict)
        spectrums = spectra if i == 0 else torch.cat(
            (spectrums, spectra), dim=0)
        paras = para_dict if i == 0 else {k: torch.cat(
            (paras[k], para_dict[k]), dim=0) for k in para_dict.keys()}
        print(f"Finished {i+1}00 samples")

    # save the sampled dataset
    df = pd.DataFrame(spectrums.cpu().numpy(), columns=S2_FULL_BANDS)
    for attr in paras.keys():
        df[attr] = paras[attr].cpu().numpy()

    # mkdir if not exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
    # save the dataset
    df.to_csv(os.path.join(SAVE_PATH, "synthetic.csv"), index=False)
    print("Done!")


if __name__ == "__main__":
    run_sampling()
