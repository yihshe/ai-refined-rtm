# This script will be used to test the pytorch implementation of RTM
import numpy as np
import unittest
from rtm.rtm import RTM as RTM_np
from rtm_torch.rtm import RTM as RTM_torch
import re
import sys
import IPython

rtm_np = RTM_np()
rtm_torch = RTM_torch()
# set a random seed for numpy and torch
np.random.seed(0)


def para_sampling(num_samples=100):
    # run uniform sampling for each variables for the test
    para_dict = {}
    # Brightness Factor (psoil) when using default soil spectrum
    # para_dict["psoil"] = np.full(num_samples, 0.8)
    para_dict["psoil"] = np.random.uniform(0.0, 1.0, num_samples)

    # Leaf Model Parameters
    # N: Structure Parameter (N)
    para_dict["N"] = np.random.uniform(1.0, 3.0, num_samples)
    # cab: Chlorophyll A+B (cab)
    para_dict["cab"] = np.random.uniform(0.0, 100.0, num_samples)
    # cw: Water Content (Cw)
    para_dict["cw"] = np.random.uniform(0.001, 0.7, num_samples)
    # cm: Dry Matter (cm)
    para_dict["cm"] = np.random.uniform(0.0001, 0.02, num_samples)
    # car: Carotenoids (Ccx)
    para_dict["car"] = np.random.uniform(0.0, 30.0, num_samples)
    # cbrown: Brown Pigments (Cbrown)
    para_dict["cbrown"] = np.random.uniform(0.0, 1.0, num_samples)
    # anth: Anthocyanins (Canth) TODO
    para_dict["anth"] = np.random.uniform(0, 5, num_samples)
    # cp: Proteins (Cp)
    para_dict["cp"] = np.random.uniform(0.0, 0.01, num_samples)
    # cbc: Carbon-based constituents (CBC)
    para_dict["cbc"] = np.random.uniform(0.0, 0.01, num_samples)

    # Canopy Model Parameters
    # LAI: (Single) Leaf Area Index (LAI)
    para_dict["LAI"] = np.random.uniform(0.01, 10.0, num_samples)
    # typeLIDF: Leaf Angle Distribution (LIDF) type: 1 = Beta, 2 = Ellipsoidal
    # if typeLIDF = 2, LIDF is set to between 0 and 90 as Leaf Angle to calculate the Ellipsoidal distribution
    # if typeLIDF = 1, LIDF is set between 0 and 5 as index of one of the six Beta distributions
    para_dict["typeLIDF"] = np.full(num_samples, 2)
    # LIDF: Leaf Angle (LIDF), only used when LIDF is Ellipsoidal
    para_dict["LIDF"] = np.random.uniform(0.0, 90.0, num_samples)
    # hspot: Hot Spot Size Parameter (Hspot)
    para_dict["hspot"] = np.random.uniform(0.0, 1.0, num_samples)
    # tto: Observation zenith angle (Tto)
    para_dict["tto"] = np.random.uniform(0.0, 89.0, num_samples)
    # tts: Sun zenith angle (Tts)
    para_dict["tts"] = np.random.uniform(0.0, 89.0, num_samples)
    # psi: Relative azimuth angle (Psi)
    para_dict["psi"] = np.random.uniform(0.0, 180.0, num_samples)

    # Forest Model Parameters
    # LAIu: Undergrowth LAI (LAIu)
    para_dict["LAIu"] = np.random.uniform(0.01, 10.0, num_samples)
    # sd: Stem Density (SD)
    para_dict["sd"] = np.random.uniform(0.0, 5000.0, num_samples)
    # h: Tree Height (H)
    para_dict["h"] = np.random.uniform(0.0, 50.0, num_samples)
    # cd: Crown Diameter (CD)
    para_dict["cd"] = np.random.uniform(0.0, 30.0, num_samples)

    return para_dict


class RTMTest(unittest.TestCase):
    def test_functionality(self, num_samples=100):
        # test if the rtm_torch is working as expected
        para_dict = para_sampling(num_samples=num_samples)
        # run the rtm_np
        rtm_np.para_reset(**para_dict)
        rtm_np.mod_exec(mode="batch")
        expected_result = rtm_np.myResult
        # run the rtm_torch
        rtm_torch.para_reset(**para_dict)
        rtm_torch.mod_exec(mode="batch")
        actual_result = rtm_torch.myResult.cpu().numpy()

        # compare the results
        self.assertEqual(actual_result.shape, expected_result.shape)
        # np.testing.assert_allclose(actual_result, expected_result, rtol=1e-5)
        max_abs_diff = 0
        mismatch_count = 0
        total_count = 0
        # percent_mismatch = 0
        try:
            np.testing.assert_allclose(
                actual_result, expected_result, atol=1e-5)
        except AssertionError as e:
            max_abs_diff_match = re.search(
                r"Max absolute difference: ([\d.e-]+)", str(e))
            max_abs_diff = float(max_abs_diff_match.group(1))
            print(f"Max Absolute Difference: {max_abs_diff}")
            mismatch = re.search(
                # r"Mismatched elements: (\d+) / (\d+) \((\d+\.\d+)%\)", str(e))
                r"Mismatched elements: (\d+) / (\d+)", str(e))
            mismatch_count = int(mismatch.group(1))
            total_count = int(mismatch.group(2))
            percent_mismatch = float(mismatch_count/total_count*100)
            print(
                f"Mismatched elements: \
                    {mismatch_count} / {total_count} ({round(percent_mismatch, 3)}%)")

        return (max_abs_diff, mismatch_count, total_count, percent_mismatch)


if __name__ == "__main__":
    # create a log file if it doesn't exist
    # sys.stdout = open("saved/log/rtm_unittest.log", "w")
    stdout = sys.stdout
    # run the test
    with open("saved/log/rtm_unittest.log", "w") as f:
        sys.stdout = f
        max_abs_diffs = []
        mismatch_counts = []
        total_counts = []
        percent_mismatches = []
        runs = 100
        num_samples = 100
        for i in range(runs):
            print(f"Running Test {i}")
            result = RTMTest().test_functionality(num_samples=num_samples)
            max_abs_diffs.append(result[0])
            mismatch_counts.append(result[1])
            total_counts.append(result[2])
            percent_mismatches.append(result[3])

        print(
            f"Max Absolute Difference over {runs*num_samples*13} elements: \
                {max(max_abs_diffs)}")
        print(f"Mismatched elements: {sum(mismatch_counts)}")
        print(f"Total elements: {sum(total_counts)}")
        print(f"Percent Mismatch: {round(sum(percent_mismatches)/runs, 3)}%")

    sys.stdout = stdout
    print("Done!")
