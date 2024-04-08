from _SeamountSupport import _SeamountSupport
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

class GaussianSupport(_SeamountSupport):
    """
    Class to support the Gausian Process Model
    """

    def __init__(self, test_data, train_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
        """
        Initializes the GaussianSupport class
        Parameters
        ----------
        test_data : str
            Path to the test data
        train_zone : array-like
            Area of that the algorithm is being trained on in the form
            [min_lat, max_lat, min_lon, max_lon]; default is the whole world
        sheet : str
            Name of the sheet in the excel file to read from
        """
        super().__init__(test_data, train_zone, sheet)
        self.end_params = None  # values of the best parameters
        self.test = False
        self.classifier = GaussianProcessClassifier(n_jobs=3)

    def fitGauss(self, data, verbose=False):
        """
        Fits a Gaussian Process Model to the data
        Parameters
        ----------
        data : array-like
            Data to fit
        verbose : bool
            Whether to print out the results
        Returns
        -------
        None
        """
        self.addTrainingData(data)
        self.classifier.fit(self.training_data[:, :-1], self.training_data[:, -1])  # type: ignore

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from LocalPath import LOCALPATH

    grav = pd.read_csv(LOCALPATH + 'data/test_curv_32.highpass.csv', names=['Longitude', 'Latitude', 'Intensity'], header=0)
    grav = grav[['Longitude', 'Latitude', 'Intensity']]
    Z = GaussianSupport.formatData(grav, 'Intensity')
    data = Z
    GausModel_test = GaussianSupport(LOCALPATH+"data/sample_mask.txt.xlsx", train_zone=(-6, -1, -98, -90))
    GausModel_test.fitGauss(data)
