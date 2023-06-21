# -*- coding: utf-8 -*-
"""
***************************************************************************
    rtm.py - Adapted from the original RTM code IVVM_GUI.py by Martin Danner
    the adapted rtm will be implemented in PyTorch to train neural networks
***************************************************************************
    IVVRM_GUI.py - LMU Agri Apps - Interactive Visualization of Vegetation Reflectance Models (IVVRM)
    -----------------------------------------------------------------------
    begin                : 01/2018
    copyright            : (C) 2018 Martin Danner; Matthias Wocher
    email                : m.wocher@lmu.de

***************************************************************************
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
                                                                                                                                                 *
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.
***************************************************************************
"""
import csv
import sys
import os
import numpy as np
from scipy.interpolate import interp1d

from rtm.Resources.PROSAIL import call_model as mod
from rtm.Resources.Spec2Sensor.Spec2Sensor_core import Spec2Sensor, BuildTrueSRF, BuildGenericSRF
from rtm import APP_DIR

import warnings
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

# class RTM is the main class to run the Radiative Transfer Model


class RTM:
    def __init__(self):
        super(RTM, self).__init__()
        # store all model choices available for the user
        self.model_choice_init()
        # initialize the model parameters
        self.para_init()
        # self.mod_exec()

    def model_choice_init(self):
        list_dir = os.listdir(APP_DIR + "/Resources/Spec2Sensor/srf")
        # Get all files in the SRF directory
        list_allfiles = [item for item in list_dir if os.path.isfile(
            APP_DIR + "/Resources/Spec2Sensor/srf/" + item)]
        # Get all files from that list with extension .srf, but pop the extension to get the name of the sensor
        list_files = [item.split('.')[0] for item in list_allfiles if item.split('.')[
            1] == 'srf']
        list_files.insert(0, 'default')

        # store sensor list
        self.sensor_list = list_files
        # store leaf model list
        self.lop_list = ['prospect4', 'prospect5', 'prospect5B', 'prospectD',
                         'prospectPro']
        # store canopy model list
        self.canopy_arch_list = ['sail', 'inform', 'None']

    def para_init(self):
        # Set all parameters to default values according to the app UI
        # specify the sensor type
        self.sensor = "Sentinel2_Full"
        # specify the leaf model type
        self.lop = "prospectD"
        # specify the canopy model type
        self.canopy_arch = "inform"

        # set the default values for the parameters of leaf and canopy models
        self.para_names = ["N", "cab", "cw", "cm", "LAI", "typeLIDF", "LIDF",
                           "hspot", "psoil", "tts", "tto", "psi", "cp", "cbc",
                           "car", "anth", "cbrown", "LAIu", "cd", "sd", "h"]
        # dictionary for parameters is initialized with Nones
        self.para_dict = dict(
            zip(self.para_names, [None] * len(self.para_names))
        )

        # Background Parameters
        # Default soil spectrum without loading background spectrum
        self.bg_type = "default"
        self.bg_spec = None
        # Brightness Factor (psoil) when using default soil spectrum
        self.para_dict["psoil"] = 0.8

        # Leaf Model Parameters
        # N: Structure Parameter (N)
        self.para_dict["N"] = 1.5
        # cab: Chlorophyll A+B (cab)
        self.para_dict["cab"] = 40
        # cw: Water Content (Cw)
        self.para_dict["cw"] = 0.03
        # cm: Dry Matter (cm)
        self.para_dict["cm"] = 0.012
        # car: Carotenoids (Ccx)
        self.para_dict["car"] = 10
        # cbrown: Brown Pigments (Cbrown)
        self.para_dict["cbrown"] = 0.25
        # anth: Anthocyanins (Canth)
        self.para_dict["anth"] = 2
        # cp: Proteins (Cp)
        self.para_dict["cp"] = 0.0015
        # cbc: Carbon-based constituents (CBC)
        self.para_dict["cbc"] = 0.01

        # Canopy Model Parameters
        # LAI: (Single) Leaf Area Index (LAI)
        self.para_dict["LAI"] = 7 if self.canopy_arch == "inform" else 3
        # typeLIDF: Leaf Angle Distribution (LIDF) type: 1 = Beta, 2 = Ellipsoidal
        # if typeLIDF = 2, LIDF is set to between 0 and 90 as Leaf Angle to calculate the Ellipsoidal distribution
        # if typeLIDF = 1, LIDF is set between 0 and 5 as index of one of the six Beta distributions
        self.para_dict["typeLIDF"] = 1
        # LIDF: Leaf Angle (LIDF), only used when LIDF is Ellipsoidal
        self.para_dict["LIDF"] = 5
        # hspot: Hot Spot Size Parameter (Hspot)
        self.para_dict["hspot"] = 0.01
        # tto: Observation zenith angle (Tto)
        self.para_dict["tto"] = 0
        # tts: Sun zenith angle (Tts)
        self.para_dict["tts"] = 30
        # psi: Relative azimuth angle (Psi)
        self.para_dict["psi"] = 0

        # Forest Model Parameters
        # LAIu: Undergrowth LAI (LAIu)
        self.para_dict["LAIu"] = 0.1
        # sd: Stem Density (SD)
        self.para_dict["sd"] = 650
        # h: Tree Height (H)
        self.para_dict["h"] = 20
        # cd: Crown Diameter (CD)
        self.para_dict["cd"] = 4.5

        # TODO set data_mean to None for future evaluations
        self.data_mean = None

        self.reset_non_learnable = True

    def select_model(self,
                     sensor="Sentinel2_Full",
                     lop="prospectD",
                     canopy_arch="inform",
                     bg_type="default",
                     bg_spec=None,):
        # Reset the choice of sub-models
        assert sensor in self.sensor_list
        assert lop in self.lop_list
        assert canopy_arch in self.canopy_arch_list
        self.sensor = sensor
        self.lop = lop
        self.canopy_arch = canopy_arch
        self.bg_type = bg_type
        self.bg_spec = bg_spec

    def para_reset(self, **para_dict):
        # TODO decide the learnable parameters to reset
        # TODO should we bound the range of parameters in model training?
        # or should the model just learn a scale factor as in Pheno-VAE?
        self.para_dict.update(para_dict)
        # print("Parameters updated!")
        if self.reset_non_learnable:
            batch_size = len(list(para_dict.values())[0])
            if batch_size > 1:
                # Reset the non-learnable parameters
                for k, v in self.para_dict.items():
                    if k not in para_dict.keys():
                        # extend the non-learnable parameters to batch size
                        self.para_dict[k] = np.full(batch_size, v)
            self.reset_non_learnable = False

    # execute the model to run the radiative transfer model
    def mod_exec(self, mode="single"):
        """
        This function is called whenever PROSAIL is to be triggered; 
        it is the function to sort all inputs and call PROSAIL with the 
        selected settings
        """
        assert mode in ["single", "batch"]
        # create new Instance of the RTM
        mod_I = mod.InitModel(
            lop=self.lop, canopy_arch=self.canopy_arch, nodat=-999,
            int_boost=1.0, s2s=self.sensor
        )
        # initialize a single model run
        # TODO decide which parameters are learnable or to be learned
        # TODO how to bound the parameters range during the learning process?
        if mode == "single":
            self.myResult = mod_I.initialize_single(soil=self.bg_spec,
                                                    tts=self.para_dict["tts"],
                                                    tto=self.para_dict["tto"],
                                                    psi=self.para_dict["psi"],
                                                    N=self.para_dict["N"],
                                                    cab=self.para_dict["cab"],
                                                    cw=self.para_dict["cw"],
                                                    cm=self.para_dict["cm"],
                                                    LAI=self.para_dict["LAI"],
                                                    LIDF=self.para_dict["LIDF"],
                                                    typeLIDF=self.para_dict["typeLIDF"],
                                                    hspot=self.para_dict["hspot"],
                                                    psoil=self.para_dict["psoil"],
                                                    cp=self.para_dict["cp"],
                                                    cbc=self.para_dict["cbc"],
                                                    car=self.para_dict["car"],
                                                    cbrown=self.para_dict["cbrown"],
                                                    anth=self.para_dict["anth"],
                                                    LAIu=self.para_dict["LAIu"],
                                                    cd=self.para_dict["cd"],
                                                    sd=self.para_dict["sd"],
                                                    h=self.para_dict["h"])[0, :]
        elif mode == "batch":
            self.myResult = mod_I.initialize_multiple_simple(soil=self.bg_spec,
                                                             tts=self.para_dict["tts"],
                                                             tto=self.para_dict["tto"],
                                                             psi=self.para_dict["psi"],
                                                             N=self.para_dict["N"],
                                                             cab=self.para_dict["cab"],
                                                             cw=self.para_dict["cw"],
                                                             cm=self.para_dict["cm"],
                                                             LAI=self.para_dict["LAI"],
                                                             LIDF=self.para_dict["LIDF"],
                                                             typeLIDF=self.para_dict["typeLIDF"],
                                                             hspot=self.para_dict["hspot"],
                                                             psoil=self.para_dict["psoil"],
                                                             cp=self.para_dict["cp"],
                                                             cbc=self.para_dict["cbc"],
                                                             car=self.para_dict["car"],
                                                             cbrown=self.para_dict["cbrown"],
                                                             anth=self.para_dict["anth"],
                                                             LAIu=self.para_dict["LAIu"],
                                                             cd=self.para_dict["cd"],
                                                             sd=self.para_dict["sd"],
                                                             h=self.para_dict["h"])
