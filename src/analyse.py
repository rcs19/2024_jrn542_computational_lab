from epic1d import *

ncells_list = arange(10,31,10)

for ncells in ncells_list:
    Run_and_Save(npart=1000, ncells=ncells)