import os
import numpy as np
import torch
import json
from rtm_torch.rtm import RTM
from rtm.rtm import RTM as RTM_np
import pandas as pd
torch.manual_seed(0)
np.random.seed(0)

SAVE_PATH = "/maps/ys611/ai-refined-rtm/data/synthetic/20240124"
# B01 and B10 will not be used in the training
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED',
                 'B05_RE1', 'B06_RE2', 'B07_RE3', 'B08_NIR1',
                 'B8A_NIR2', 'B09_WV', 'B10', 'B11_SWI1',
                 'B12_SWI2']
rtm_paras = json.load(open('/maps/ys611/ai-refined-rtm/configs/rtm_paras.json'))

def para_sampling(num_samples=100):
    # run uniform sampling for learnable parameters
    para_dict = {}
    for para_name in rtm_paras.keys():
        min = rtm_paras[para_name]['min']
        max = rtm_paras[para_name]['max']
        para_dict[para_name] = torch.rand(num_samples) * (max - min) + min
    SD = 500
    para_dict['cd'] = torch.sqrt(
        (para_dict['fc']*10000)/(torch.pi*SD))*2
    para_dict['h'] = torch.exp(
        2.117 + 0.507*torch.log(para_dict['cd']))
    
    return para_dict

def run_sampling(sampling_np=False):
    # sample the dataset and save it to a csv file
    rtm = RTM()
    if sampling_np:
        rtm_np = RTM_np()
    for i in range(180):
        para_dict = para_sampling(num_samples=100)
        with torch.no_grad():
            spectrum = rtm.run(**para_dict)
        # concatenate the spectra
        spectra = spectrum if i == 0 else torch.cat(
            (spectra, spectrum), dim=0)
        # save only the parameters in their original scales
        paras = para_dict if i == 0 else {k: torch.cat(
            (paras[k], para_dict[k]), dim=0) for k in para_dict.keys()}
    
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
    df = pd.DataFrame(spectra.cpu().numpy(), columns=S2_FULL_BANDS)
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
