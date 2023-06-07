"""
exposure_response_function.py
=============================
The core module for the ExposureResponseFunction class.
"""

from pycausalgps.erf_helper import estimate_npmetric_erf
from pycausalgps.rscripts.rfunctions import (estimate_pmetric_erf, 
                                             estimate_semipmetric_erf)




class ExposureResponseFunction:
    """
    The ExposureResponseFunction class is the core class for computing the
    exposure response function. Three types of exposure response functions are
    supported: parametric, semiparametric, and nonparametric. 
    """



    ERF_PARAMETRIC = "parametric"
    ERF_SEMIPARAMETRIC = "semiparametric"
    ERF_NONPARAMETRIC = "nonparametric"


    def __init__(self, data, params) -> None:
        self.data = data
        self.params = params
        self.outcome = {}
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

        erf_type = self.params.get("erf_type")

        if erf_type == self.ERF_PARAMETRIC:
            output_obj = self._erf_parametric()
            self.outcome["erf_type"] = erf_type
            self.outcome["erf_data"] = output_obj.get("data")   
        elif erf_type == self.ERF_SEMIPARAMETRIC:
            output_obj = self._erf_semiparametric()
            self.outcome["erf_type"] = erf_type
            self.outcome["erf_data"] = output_obj.get("data")
        elif erf_type == self.ERF_NONPARAMETRIC:
            output_obj = self._erf_nonparametric()
        else:
            raise Exception(f"The provided erf_type ({erf_type}) is not " 
                            f"supported. Available options: "
                            f"parametric, semiparametric, or nonparametric.")

    def _erf_parametric(self):

        if "formula" not in self.params:
            raise Exception("formula is required for parametric erf.")
        
        if "family" not in self.params:
            raise Exception("family is required for parametric erf.")

        result = estimate_pmetric_erf(self.params.get("formula"),
                                      self.params.get("family"),
                                      self.data)
        output_obj = {"data": result}
        return output_obj

    def _erf_semiparametric(self):

        if "formula" not in self.params:
            raise Exception("formula is required for semiparametric erf.")
        
        if "family" not in self.params:
            raise Exception("family is required for semiparametric erf.")
        
        result = estimate_semipmetric_erf(self.params.get("formula"),
                                          self.params.get("family"),
                                          self.data)
        output_obj = {"data": result}
        return output_obj
        

    def _erf_nonparametric(self):

        if "bw_seq" not in self.params:
            raise Exception("bw_seq is required for nonparametric erf.")
        
        if "w_vals" not in self.params:
            raise Exception("w_vals is required for nonparametric erf.")
        
        m_Y = self.data.get("Y").to_numpy()
        m_w = self.data.get("w").to_numpy()
        counter_weight = self.data.get("counter_weight").to_numpy()
        bw_seq = self.params.get("bw_seq")
        w_vals = self.params.get("w_vals")

        if "nthread" not in self.params:
            nthread = 1
        else:
            nthread = self.params.get("nthread")

        results = estimate_npmetric_erf(m_Y, 
                                        m_w, 
                                        counter_weight, 
                                        bw_seq, 
                                        w_vals, 
                                        nthread)
        
        return results


