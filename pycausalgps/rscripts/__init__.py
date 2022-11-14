
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


# import R's "utils" package
utils = importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# define packages to be installed
packnames = ('polycor', 'locpol', 'wCorr')

# Install packages if not installed
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]

if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

