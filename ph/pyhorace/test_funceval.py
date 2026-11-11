from pyhorace import *
from pyhorace.data.graphics import *
from pyhorace.coord_transf import *
from pyhorace.models import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("QtAgg")
plt.ion();
hp = hor_config();
hp.mem_chunk_size = 10000000

w4 = sqw(r'e:\SHARE\Fe\Data\sqw\sqw2024\Fe_ei200_align_sym3.sqw')

lp = line_proj()
w1 = cut(w4,lp,[0.02],[-0.1,0.1],[-0.1,0.1],[70,80])
acolor('k')
w1.plot();
#plt.show();


#init_fg_params = [T,gamma,Seff, gap, J0, J1, J2, J3, J4];
in_par = np.array([8,20,50,0,35])
w1_eval = w1.sqw_eval(sqw_bcc_hfm,in_par)
acolor('g')
w1_eval.pl()
plt.show()