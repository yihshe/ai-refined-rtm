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

# from enmapbox.gui.utils import loadUi
# from PyQt5 import uic

pathUI_IVVRM = os.path.join(APP_DIR, 'Resources/UserInterfaces/IVVRM_main.ui')
pathUI_loadtxt = os.path.join(
    APP_DIR, 'Resources/UserInterfaces/LoadTxtFile.ui')
pathUI_wavelengths = os.path.join(
    APP_DIR, 'Resources/UserInterfaces/Select_Wavelengths.ui')
pathUI_sensor = os.path.join(
    APP_DIR, 'Resources/UserInterfaces/GUI_SensorEditor.ui')

# class RTM is the main class to run the Radiative Transfer Model


class RTM:
    def __init__(self):
        super(RTM, self).__init__()
        # NOTE whether to use the GUI parameters or not is not yet decided
        # uic.loadUi(pathUI_IVVRM, self)

        # self.special_chars()    # place special characters that could not be set in Qt Designer
        self.initial_values()
        # self.update_slider_pos()
        # self.update_lineEdit_pos()
        self.deactivate_sliders()
        self.init_sensorlist()
        self.para_init()
        self.select_model()
        self.mod_interactive()
        self.mod_exec()

    def initial_values(self):
        # TODO currently all the parameters are set to default values
        self.lop = "prospectD"
        self.canopy_arch = "sail"
        self.para_names = ["N", "cab", "cw", "cm", "LAI", "typeLIDF", "LIDF",
                           "hspot", "psoil", "tts", "tto", "psi", "cp", "cbc",
                           "car", "anth", "cbrown", "LAIu", "cd", "sd", "h"]

        self.data_mean = None
        # dictionary for parameters is initialized with Nones
        self.para_dict = dict(
            zip(self.para_names, [None] * len(self.para_names))
        )
        self.bg_spec = None
        self.bg_type = "default"

    def init_sensorlist(self):
        # TODO currently the sensor list is not yet implemented
        pass

    def select_s2s(self):
        # TODO currently the sensor list is not yet implemented
        pass

    def para_init(self):
        # TODO check the implementation of select_s2s
        # initialize the sensor without triggering a new PROSAIL run
        self.select_s2s(sensor_index=0, trigger=False)
        # TODO set the default values for the parameters
        self.para_dict["N"] = 1.5
        self.para_dict["cab"] = 40
        self.para_dict["cw"] = 0.03
        self.para_dict["cm"] = 0.012
        self.para_dict["LAI"] = 2
        self.para_dict["typeLIDF"] = 2
        self.para_dict["LIDF"] = 0.5
        self.para_dict["hspot"] = 0.01
        self.para_dict["psoil"] = 0.5
        self.para_dict["tts"] = 45
        self.para_dict["tto"] = 45
        self.para_dict["psi"] = 0
        self.para_dict["cp"] = 0
        self.para_dict["car"] = 0.01
        self.para_dict["anth"] = 0.01
        self.para_dict["cbrown"] = 0.01
        self.para_dict["LAIu"] = 0
        self.para_dict["cd"] = 0.01
        self.para_dict["sd"] = 0.01
        self.para_dict["h"] = 0.01
        self.para_dict["cbc"] = 0.01

    # execute the model to run the radiative transfer model
    def mod_exec(self):
        """
        This function is called whenever PROSAIL is to be triggered; 
        it is the function to sort all inputs and call PROSAIL with the 
        selected settings
        """
        # create new Instance of the RTM
        mod_I = mod.InitModel(
            lop=self.lop, canopy_arch=self.canopy_arch, nodat=-999,
            int_boost=1.0, s2s=self.sensor
        )
        # initialize a single model run
        # TODO decide which parameters are learnable or to be learned
        # TODO how to bound the parameters range during the learning process?
        self.myResult = mod_I.initialize_single(tts=self.para_dict["tts"],
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
                                                soil=self.bg_spec,
                                                LAIu=self.para_dict["LAIu"],
                                                cd=self.para_dict["cd"],
                                                sd=self.para_dict["sd"],
                                                h=self.para_dict["h"])[0, :]

        # NOTE plotting may not be necessary but track the output during training
        self.plotting()

        def plotting(self):
            # TODO implement plotting
            pass
