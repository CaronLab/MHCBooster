import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def draw_venn_diagram(set1, set2, set3, labels=('Set1', 'Set2', 'Set3'), title='Venn Diagram'):
    plt.figure(figsize=(4, 4))
    venn3([set(set1), set(set2), set(set3)], labels)
    plt.title('RA_Fractionation')
    plt.show()

perc = 'RLATVLPGLEV,EAIMADEAL,MPYMPPASDRL,EVFGYDAYSALPR,VTLNPDLYV,FLAFFLDLI,DTQGDELLLALPR,FVFATPTLGLTVK'
frag = 'YPSSPVFVI,FPLDLRTLL,KILDRIVFL,VPLFTAIAL,IPAAVQAL,MPYMPPASDRL,EVFGYDAYSALPR,RLATVLPGLEV,FGAEDNEVF,GPWVPEQWM,FVFATPTLGLTVK,DTQGDELLLALPR,FLAFFLDLI,VTLNPDLYV'
mhcb = 'NYDLLRLEL,FPLDLRTLL,YPSSPVFVI,EVFGYDAYSALPR,KILDRIVFL,MPYMPPASDRL,EAIMADEAL,IPAAVQAL,IPAVSVPIL,VPLFTAIAL,IVAPYLFWL,RLATVLPGLEV,YLLEMLWRL,LLVDLLWLL,DTQGDELLLALPR,FVFATPTLGLTVK,GPWVPEQWM,VTLNPDLYV,FLAFFLDLI'

draw_venn_diagram(set(perc.split(',')), set(frag.split(',')), set(mhcb.split(',')), labels=['Percolator', 'FragPipe', 'MHCBooster'])