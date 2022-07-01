if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph #get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1
y.backward()

# 변수 이름 지정
x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

# txt = get_dot_graph(y, verbose=False)
# print(txt)

# with open('sample.dot', 'w') as o:
#     o.write(txt)
plot_dot_graph(y, verbose=False, to_file='test.png')