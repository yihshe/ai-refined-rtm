# -*- coding: utf-8 -*-
"""
***************************************************************************
    call_model.py
    -----------------------------------------------------------------------
    begin                : 09/2020
    copyright            : (C) 2020 Martin Danner; Matthias Wocher
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
    call_model.py routine handles the execution of PROSAIL in different forms
    This version is vectorized, i.e. the idea is to pass several parameter inputs at once to obtain an array of results
    But it is also used for single outputs

"""

import os
import torch
import numpy as np
from scipy.stats import truncnorm
import rtm_torch.Resources.PROSAIL.SAIL as SAIL_v
import rtm_torch.Resources.PROSAIL.INFORM as INFORM_v
import rtm_torch.Resources.PROSAIL.prospect as prospect_v
from rtm_torch.Resources.PROSAIL.dataSpec import lambd
from rtm_torch.Resources.Spec2Sensor.Spec2Sensor_core import Spec2Sensor
import warnings
import time
import math

# do not show warnings (set to 'all' if you want to see warnings, too)
warnings.filterwarnings('ignore')

# This class creates instances of the actual models and is fed with parameter inputs


class CallModel:

    def __init__(self, soil, paras):
        # paras is a dictionary of all input parameters; this allows flexible adding/removing for new models
        self.par = paras
        # cab is always part of self.par, so it is used to obtain ninputs
        self.ninputs = self.par['cab'].shape[0]
        self.soil = soil

    def call_prospect4(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_4(
            self.par["N"], self.par["cab"], self.par["cw"], self.par["cm"])

        return self.prospect

    def call_prospect5(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_5(
            self.par["N"], self.par["cab"], self.par["car"], self.par["cw"], self.par["cm"])

        return self.prospect

    def call_prospect5b(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_5B(
            self.par["N"], self.par["cab"], self.par["car"], self.par["cbrown"], self.par["cw"], self.par["cm"])

        return self.prospect

    def call_prospectD(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_D(
            self.par["N"], self.par["cab"], self.par["car"], self.par["anth"], self.par["cbrown"], self.par["cw"], self.par["cm"])
        return self.prospect

    def call_prospectPro(self):
        prospect_instance = prospect_v.Prospect()
        self.prospect = prospect_instance.prospect_Pro(
            self.par["N"], self.par["cab"], self.par["car"], self.par["anth"], self.par["cp"], self.par["cbc"], self.par["cbrown"], self.par["cw"])
        return self.prospect

    def call_4sail(self):
        try:
            self.prospect.any()  # 4sail can only be called when PROSAIL has been run first
        except ValueError:
            raise ValueError(
                "A leaf optical properties model needs to be run first!")

        # Convert degrees to radians using PyTorch
        tts_rad = self.par["tts"] * (math.pi / 180)
        tto_rad = self.par["tto"] * (math.pi / 180)
        psi_rad = self.par["psi"] * (math.pi / 180)

        # Create Instance of SAIL and initialize angles
        sail_instance = SAIL_v.Sail(tts_rad, tto_rad, psi_rad)

        self.sail = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2], self.par["LIDF"],
                                           self.par["typeLIDF"], self.par["LAI"], self.par["hspot"], self.par["psoil"],
                                           self.soil)  # call 4SAIL from the SAIL instance

        return self.sail

    def call_inform(self):
        try:
            self.prospect.any()
        except ValueError:
            raise ValueError(
                "A leaf optical properties model needs to be run first!")

        # Convert degrees to radians
        tts_rad = self.par["tts"] * (math.pi / 180)
        tto_rad = self.par["tto"] * (math.pi / 180)
        psi_rad = self.par["psi"] * (math.pi / 180)

        sail_instance = SAIL_v.Sail(tts_rad, tto_rad, psi_rad)

        # Step 1: call Pro4sail to calculate understory reflectance
        self.sail_understory_refl = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                           self.par["LIDF"], self.par["typeLIDF"],
                                                           self.par["LAIu"], self.par["hspot"],
                                                           self.par["psoil"], self.soil)

        # Step 2: call Pro4sail with understory as soil to calculate infinite crown reflectance
        # vectorized: intialize extreme LAI ninputs times
        inform_temp_LAI = torch.full(
            (self.ninputs,), 15, dtype=torch.float32).T
        # vectorized: initialize hspot = 0 ninputs times
        inform_temp_hspot = torch.full(
            (self.ninputs,), 0, dtype=torch.float32).T

        self.sail_inf_refl = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                    self.par["LIDF"], self.par["typeLIDF"],
                                                    inform_temp_LAI, self.par["hspot"],
                                                    psoil=None, soil=None, understory=self.sail_understory_refl)

        self.sail_tts_trans = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                     self.par["LIDF"], self.par["typeLIDF"],
                                                     self.par["LAI"], inform_temp_hspot,
                                                     psoil=None, soil=None, understory=self.sail_understory_refl,
                                                     inform_trans='tts')

        self.sail_tto_trans = sail_instance.pro4sail(self.prospect[:, :, 1], self.prospect[:, :, 2],
                                                     self.par["LIDF"], self.par["typeLIDF"],
                                                     self.par["LAI"], inform_temp_hspot,
                                                     psoil=None, soil=None, understory=self.sail_understory_refl,
                                                     inform_trans='tto')

        inform_instance = INFORM_v.INFORM(
            sail_instance.costts, sail_instance.costto, sail_instance.cospsi)

        inform = inform_instance.inform(self.par["cd"], self.par["sd"], self.par["h"],
                                        self.sail_understory_refl, self.sail_inf_refl,
                                        self.sail_tts_trans, self.sail_tto_trans)

        return inform
