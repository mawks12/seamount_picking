"""
Library for testing the SMV algorithm
"""

import pandas as pd
import numpy as np
from sklearn.svm import *  # TODO: import the correct SVM class
from sklearn.model_selection import train_test_split
from _SeamountSupport import _SeamountSupport

class SVMSupport(_SeamountSupport):
    """
    Class contining tester functions for the SVM algorithm
    built on the _SeamountSupport class, and contains the same
    basic functions as the DBSCANSupport class
    """

    def __init__(self, validation_data, fast=False, train_zone=..., sheet: str = "new mask") -> None:
        super().__init__(validation_data, train_zone, sheet)
        raise NotImplementedError("This class has not been implemented yet")

    def scoreTestData(self, path) -> float:
        return super().scoreTestData(path)  # TODO: implement this function
