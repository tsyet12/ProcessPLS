from processPLS.model import *
from processPLS.datasets import *


if __name__=="__main__":
  X,Y,matrix=ValdeLoirData()

  import matplotlib.pyplot as plt
  model = ProcessPLS()
  model.fit(X,Y,matrix)


  model.plot()
  plt.show()