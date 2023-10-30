'''
Material library dictrionary

Format:
{
    'material key' : [eps_0, mu_0],
}

* 'material key' in lower case only
'''
import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0

material_lib = {
    'pec' : [np.inf, mu_0],
    'vacuum' : [eps_0, mu_0]

}