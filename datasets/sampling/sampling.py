import os
import numpy as np
import torch
from rtm_torch.rtm import RTM
from rtm.rtm import RTM as RTM_np
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)

SAVE_PATH = "/maps/ys611/ai-refined-rtm/data/synthetic/20230816"
# B01 and B10 will not be used in the training
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                 'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                 'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                 'B12_SWI2']


def uniform_sampling(min, max, num_samples):
    # uniform sampling using pytorch
    para_norm = torch.rand(num_samples)
    para = para_norm * (max - min) + min
    return para, para_norm
    # return torch.rand(num_samples) * (max - min) + min


def para_sampling(num_samples=100):
    # run uniform sampling for learnable parameters
    para_dict = {}
    para_norm_dict = {}
    # # Brightness Factor (psoil) when using default soil spectrum
    # para_dict["psoil"] = np.random.uniform(0.0, 1.0, num_samples)

    # Leaf Model Parameters
    # # NOTE N: Structure Parameter (N)
    # para_dict["N"] = uniform_sampling(1.0, 4.0, num_samples)
    # # NOTE cab: Chlorophyll A+B (cab)
    # para_dict["cab"] = uniform_sampling(0.0, 100.0, num_samples)
    # # NOTE cw: Water Content (Cw)
    # para_dict["cw"] = uniform_sampling(0.0002, 0.08, num_samples)
    # # NOTE cm: Dry Matter (cm)
    # para_dict["cm"] = uniform_sampling(0.0, 0.05, num_samples)
    # NOTE N: Structure Parameter (N)
    para_dict["N"], para_norm_dict["N"] = uniform_sampling(
        1.0, 3.0, num_samples)
    # NOTE cab: Chlorophyll A+B (cab)
    para_dict["cab"], para_norm_dict["cab"] = uniform_sampling(
        10.0, 80.0, num_samples)
    # NOTE cw: Water Content (Cw)
    para_dict["cw"], para_norm_dict["cw"] = uniform_sampling(
        0.001, 0.02, num_samples)
    # NOTE cm: Dry Matter (cm)
    para_dict["cm"], para_norm_dict["cm"] = uniform_sampling(
        0.005, 0.05, num_samples)

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
    # NOTE LAI: (Single) Leaf Area Index (LAI)
    # NOTE output of current RTM is not sensitive to LAI higher than 3
    # para_dict["LAI"] = uniform_sampling(0.01, 15.0, num_samples)
    para_dict["LAI"], para_norm_dict["LAI"] = uniform_sampling(
        0.01, 5.0, num_samples)
    # # typeLIDF: Leaf Angle Distribution (LIDF) type: 1 = Beta, 2 = Ellipsoidal
    # # if typeLIDF = 2, LIDF is set to between 0 and 90 as Leaf Angle to calculate the Ellipsoidal distribution
    # # if typeLIDF = 1, LIDF is set between 0 and 5 as index of one of the six Beta distributions
    # para_dict["typeLIDF"] = np.full(num_samples, 2)
    # LIDF: Leaf Angle (LIDF), only used when LIDF is Ellipsoidal
    # para_dict["LIDF"] = uniform_sampling(0.0, 90.0, num_samples)
    # # hspot: Hot Spot Size Parameter (Hspot)
    # para_dict["hspot"] = np.random.uniform(0.0, 1.0, num_samples)
    # # tto: Observation zenith angle (Tto)
    # para_dict["tto"] = np.random.uniform(0.0, 89.0, num_samples)
    # # tts: Sun zenith angle (Tts)
    # para_dict["tts"] = np.random.uniform(0.0, 89.0, num_samples)
    # # psi: Relative azimuth angle (Psi)
    # para_dict["psi"] = np.random.uniform(0.0, 180.0, num_samples)

    # Forest Model Parameters
    # # NOTE LAIu: Undergrowth LAI (LAIu)
    # para_dict["LAIu"] = uniform_sampling(0.01, 3.0, num_samples)
    # # NOTE sd: Stem Density (SD)
    # para_dict["sd"] = uniform_sampling(0.0, 3000.0, num_samples)
    # # NOTE h: Tree Height (H)
    # para_dict["h"] = uniform_sampling(1.0, 50.0, num_samples)
    # # NOTE cd: Crown Diameter (CD)
    # para_dict["cd"] = uniform_sampling(1.0, 15.0, num_samples)
    # NOTE LAIu: Undergrowth LAI (LAIu)
    para_dict["LAIu"], para_norm_dict["LAIu"] = uniform_sampling(
        0.01, 1.0, num_samples)
    # NOTE sd: Stem Density (SD)
    para_dict["sd"], para_norm_dict["sd"] = uniform_sampling(
        100, 5000, num_samples)
    # para_dict["sd"], para_norm_dict["sd"] = uniform_sampling(
    #     0.0, 628.0, num_samples)  # corresponding to cd = 4.5
    # NOTE h: Tree Height (H)
    para_dict["h"], para_norm_dict["h"] = uniform_sampling(
        1.0, 15.0, num_samples)
    # NOTE cd: Crown Diameter (CD)
    # para_dict["cd"], para_norm_dict["cd"] = uniform_sampling(
    #     1.0, 5.0, num_samples)
    # calculate cd from sampled sd and fractional coverage fc
    fc, para_norm_dict["fc"] = uniform_sampling(0.1, 1, num_samples)
    para_dict["cd"] = torch.sqrt((fc*10000)/(torch.pi*para_dict["sd"]))*2
    return para_dict, para_norm_dict


def cd(sd):
    # calculate the crown diameter given stem density using tensor
    # NOTE: as start just try the simple method to calculate cd
    # first samle the coverage
    cd = torch.sqrt(10000/(torch.pi*sd))*2
    return cd


def sd(cd):
    # calculate the stem density given crown diameter using tensor
    sd = 10000/(torch.pow(cd/2, 2)*torch.pi)
    return sd


def coverage(cd, sd):
    # calculate the crown coverage given stem density using tensor
    return torch.pow(cd/2, 2)*torch.pi*sd


def run_sampling(sampling_np=False):
    # sample the dataset and save it to a csv file
    rtm = RTM()
    if sampling_np:
        rtm_np = RTM_np()
    for i in range(180):
        para_dict, para_norm_dict = para_sampling(num_samples=100)
        # replace the cd with the calculated value
        # cov = coverage(para_dict["cd"], para_dict["sd"])
        # para_dict["cd"][cov > 10000] = cd(para_dict["sd"])[cov > 10000]
        # para_dict["cd"] = cd(para_dict["sd"])
        # para_dict["sd"][cov > 10000] = sd(para_dict["cd"])[cov > 10000]
        # run the RTM without tracking gradients
        with torch.no_grad():
            spectra = rtm.run(**para_dict)

        # NOTE additional filter to make sure the coverage is smaller than 10000
        # cov = coverage(para_dict["cd"], para_dict["sd"])
        # para_dict = {k: v[cov <= 10000] for k, v in para_dict.items()}
        # para_norm_dict = {k: v[cov <= 10000]
        #                   for k, v in para_norm_dict.items()}
        # spectra = spectra[cov <= 10000]

        spectrums = spectra if i == 0 else torch.cat(
            (spectrums, spectra), dim=0)
        # NOTE choose to save para or para_norm
        # paras = para_dict if i == 0 else {k: torch.cat(
        #     (paras[k], para_dict[k]), dim=0) for k in para_dict.keys()}
        paras = para_norm_dict if i == 0 else {k: torch.cat(
            (paras[k], para_norm_dict[k]), dim=0) for k in para_norm_dict.keys()}

        if sampling_np:
            # run the RTM_np
            para_dict_np = {k: v.cpu().numpy() for k, v in para_dict.items()}
            rtm_np.para_reset(**para_dict_np)
            rtm_np.mod_exec(mode="batch")
            spectrums_np = rtm_np.myResult if i == 0 else np.concatenate(
                (spectrums_np, rtm_np.myResult), axis=0)

        print(f"Finished {i+1}00 samples")

    # mkdir if not exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)

    # save the sampled dataset
    df = pd.DataFrame(spectrums.cpu().numpy(), columns=S2_FULL_BANDS)
    for attr in paras.keys():
        df[attr] = paras[attr].cpu().numpy()
    # save the dataset
    df.to_csv(os.path.join(
        SAVE_PATH, "synthetic.csv"), index=False)

    if sampling_np:
        df_np = pd.DataFrame(spectrums_np, columns=S2_FULL_BANDS)
        for attr in paras.keys():
            df_np[attr] = paras[attr].cpu().numpy()
        df_np.to_csv(os.path.join(
            SAVE_PATH, "synthetic_np.csv"), index=False)

    print("Done!")


if __name__ == "__main__":
    run_sampling()
