import pkg_resources
import pandas as pd
def ValdeLoirData(original=False):
    stream = pkg_resources.resource_stream(__name__, '/data/ValdeLoirData.csv')
    data=pd.read_csv(stream, encoding='utf-8')
    data=data.drop(columns=data.columns[0])
    if original==False:
      smell_at_rest=data.iloc[:,:5]
      view=data.iloc[:,5:8]
      smell_after_shaking=data.iloc[:,8:18]
      tasting=data.iloc[:,18:27]
      global_quality=data.iloc[:,27]
  
      X={
      'Smell at Rest':smell_at_rest,
      "View":view,
      "Smell after Shaking":smell_after_shaking,
      "Tasting":tasting,
      }

      Y={"Global Quality":global_quality}
      matrix = pd.DataFrame(
      [
      [0,0,0,0,0], 
      [1,0,0,0,0],
      [1,1,0,0,0],
      [1,1,1,0,0],
      [1,1,1,1,0],
      ],
      index=list(X.keys())+list(Y.keys()),
      columns=list(X.keys())+list(Y.keys())
      )
      return X,Y, matrix
    else:
      return data


def OilDistillationData(original=False):

    stream = pkg_resources.resource_filename(__name__, '/data/OilDistillationData.csv')
    data=pd.read_csv(stream, encoding='utf-8',header=None)
    
    if original==False:
      feed=data.iloc[:,:4]
      preheater1=data.iloc[:,4:8]
      preheater2=data.iloc[:,8:12]
      separation=data.iloc[:,12:16]
      yield_=data.iloc[:,16]
  
      X={
      'Feed':feed,
      "Preheater I":preheater1,
      "Preheater II":preheater2,
      "Separation":separation,
      }

      Y={"Yield":yield_}
      matrix = pd.DataFrame(
      [
      [0,0,0,0,0], 
      [1,0,0,0,0],
      [0,1,0,0,0],
      [0,0,1,0,0],
      [1,1,1,1,0],
      ],
      index=list(X.keys())+list(Y.keys()),
      columns=list(X.keys())+list(Y.keys())
      )
      return X,Y, matrix
    else:
      return data


if __name__=="__main__":   
  from model import ProcessPLS
  import networkx as nx
  import math
  import matplotlib.pyplot as plt
  X,Y,M=OilDistillationData()
  print(M.T)
  '''
  G=nx.from_pandas_adjacency(M.T,create_using=nx.DiGraph)
  label_options = {"ec": "k", "fc": "white", "alpha": 0.5}
  
  nx.draw(G, pos=nx.shell_layout(G,rotate=math.pi/len(G)*(len(G)/2)), node_color='black',alpha=0.9,node_shape='o', edge_color='black', with_labels = True,node_size=3000,bbox=label_options,font_size=12)
  plt.show()'''
  model=ProcessPLS(max_lv=1)
  model.fit(X,Y,M)
  model.plot()
  plt.show()

