"""
exposure_response_function.py
=============================
The core module for the ExposureResponseFunction class.
"""




class ExposureResponseFunction:



    ERF_PARAMETRIC = "parametric"
    ERF_SEMIPARAMETRIC = "semiparametric"
    ERF_NONPARAMETRIC = "nonparametric"


    def __init__(self, data, params) -> None:
        self.data = data
        self.params = params
        self.compute_erf()
    

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        self._params = params

    
    def compute_erf(self):

        # convert the follwoing to a switch statement
 

        if self.params.get("erf_type") == self.ERF_PARAMETRIC:
            self._erf_parametric()
        elif self.params.get("erf_type") == self.ERF_SEMIPARAMETRIC:
            self._erf_semiparametric()
        elif self.params.get("erf_type") == self.ERF_NONPARAMETRIC:
            self._erf_nonparametric()
        else:
            raise Exception("erf_type must be one of parametric, semiparametric, or nonparametric.")



    def _erf_parametric(self):
        pass

    def _erf_semiparametric(self):
        pass

    def _erf_nonparametric(self):
        pass

