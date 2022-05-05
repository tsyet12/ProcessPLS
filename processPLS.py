from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, RepeatedKFold
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


def svd_signstable(X):
    try:
        X=np.asarray(X)
    except:
        pass    
    U, D, V= np.linalg.svd(X,full_matrices=False)
    V=V.T  #python V is transposed compared to matlab

    K=len(D)
    s_left=np.empty((1,K))
    
    #step 1
    for k in range(K):
        select=np.setdiff1d(list(range(K)),k)
        DD=np.zeros((K-1,K-1))
        np.fill_diagonal(DD,D[select])
        Y=X-U[:,select]@DD@V[:,select].T
        
        s_left_parts= np.empty((1,Y.shape[1]))
        
        for j in range(Y.shape[1]):
            temp_prod=(U[:,k].T)@(Y[:,j])
            s_left_parts[:,j]=np.sign(temp_prod)*(temp_prod**2)
        
        s_left[:,k]=np.sum(s_left_parts)
        
    #step 2
    s_right=np.empty((1,K)) 
    for k in range(K):
        select=np.setdiff1d(list(range(K)),k)
        DD=np.zeros((K-1,K-1))
        np.fill_diagonal(DD,D[select])
        Y=X-U[:,select]@DD@V[:,select].T
        
        s_right_parts=np.empty((1,Y.shape[0]))
        for i in range(Y.shape[0]):
            temp_prod= (V[:,k].T)@(Y[i,:].T)
            s_right_parts[:,i]=np.sign(temp_prod)*(temp_prod**2)
        s_right[:,k]=np.sum(s_right_parts)    

    #step 3
    for k in range(K):
        if (s_right[:,k]*s_right[:,k])<0:
            if s_left[:,k]<s_right[:,k]:
                s_left[:,k]=-s_left[:,k]
            else:
                s_right[:,k]=-s_right[:,k]
    left=np.zeros((K,K))
    right=np.zeros((K,K))
    np.fill_diagonal(left,np.sign(s_left))
    np.fill_diagonal(right,np.sign(s_right))
    U=U@left
    V=V@right
    return U, D, V
            
class SIMPLS(BaseEstimator):
    def __init__(self,n_components=1):
      self.__name__='SIMPLS'
      self.n_components=n_components
      self.x_weights_=None #Wx
      self.y_weights_=None#Wy
      self.x_loadings_=None#P
      self.y_loadings_=None #Q
      self.x_scores_=None #T
      self.y_scores_=None #U
      self.coef_=None  #B
      self.var_=None #V
    def fit(self, X,y):
      try:
        X=np.asarray(X)
      except:
        pass
      try:
        y=np.asarray(y)
      except:
        pass
      if y.ndim<2:
        y=y.reshape(-1,1)
      #Fun stuff
      self.x_weights_=np.zeros((X.shape[1],self.n_components)) #Wx
      self.y_weights_=np.zeros((y.shape[1],self.n_components)) #Wy
      self.x_loadings_=np.zeros((X.shape[1],self.n_components)) #P
      self.y_loadings_=np.zeros((y.shape[1],self.n_components)) #Q
      self.x_scores_=np.zeros((X.shape[0],self.n_components)) #T
      self.y_scores_=np.zeros((y.shape[0],self.n_components)) #U
      self.coef_=None  #B
      self.B0=None #B0
      self.var_=np.zeros((X.shape[1],self.n_components)) #V
      self.xfrac_var_=None
      self.yfrac_var_=None
      S=X.T@y  #Covariance matrix
      
      for i in range(self.n_components): #loop through the latent variables      
        _,_,q=svd_signstable(S) #solve the sign stable svd

        q=q[:,0]
         
        r=S@q
        
        t=X@r
        
        normt=np.linalg.norm(t)
        t=np.divide(t,normt)
        p=X.T@t
        self.x_loadings_[:,i]=p
        q=y.T@t
        self.y_loadings_[:,i]=q
        r=np.divide(r,normt)
        
        # scores and weights
        self.x_scores_[:,i]=t
        self.y_scores_[:,i]=(y@q)
        self.x_weights_[:,i]=r

        #update orthonormal basis
        vi=self.x_loadings_[:,i]

        for repeat in range(2):
            for j in range(i-1):
                vj=self.var_[:,j]
                vf=vj.T@vi
                vi=vi-(vj.T@vi)*vj
                
        vi=np.divide(vi,np.linalg.norm(vi))
        self.var_[:,i]=vi.reshape(-1,)
        
        #deflate coriance matrix
        vi=vi.reshape(-1,1)
        S=S-vi@(vi.T@S)
        Vi=self.var_[:,:i]
        S=S-Vi@(Vi.T@S)
        
        #orthogonalize Y scores to preceeding X scores
        if i>1:
            self.y_scores_[:,i]=self.y_scores_[:,i]-self.x_scores_@(self.x_scores_.T@self.y_scores_[:,i])
      #calculate B regression vector      
      self.coef_=self.x_weights_@self.y_loadings_.T
      self.B0=np.mean(y,axis=0)-np.mean(X,axis=0)@self.coef_
      #self.coef_=np.vstack((np.mean(y,axis=0)-np.mean(X,axis=0)@self.coef_,self.coef_))
      
      #calculate fraction of explained variance
      self.xfrac_var_=np.divide(np.sum(np.abs(self.x_loadings_**2),axis=0),np.sum(np.sum(np.abs(X)**2,axis=0)))
      self.yfrac_var_=np.divide(np.sum(np.abs(self.y_loadings_**2),axis=0),np.sum(np.sum(np.abs(y)**2,axis=0)))
      self.var_=np.vstack((self.xfrac_var_,self.yfrac_var_))
      
      #calculate y weights
      self.y_weights_=y.T@self.y_scores_
      return self
    def predict(self,X,y=None):
      try:
        X=np.asarray(X)
      except:
        pass
        
      y_pred=X@self.coef_+self.B0
      return y_pred
    
      

class PartialLeastSquaresCV(BaseEstimator):
  def __init__(self,cv=RepeatedKFold(10,5,random_state=999),scoring='neg_mean_squared_error',epsilon=0.0001,max_lv=30):
      self.__name__='PartialLeastSquaresCV'
      self.cv=cv
      self.model=None
      self.scoring=scoring
      self.epsilon=epsilon
      self.max_lv=max_lv
      
      self.n_components=None
      self.x_weights_=None
      self.y_weights_=None
      self.x_loadings_=None
      self.y_loadings_=None
      self.x_scores_,self.y_scores_=None , None
      self.xfrac_var_= None
      self.yfrac_var_=None
      self.coef_=None
  def predict(self, X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return self.model.predict(X)
    
  def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
        y=pd.DataFrame(y)
      except:
        pass
      try:
        y=y.to_frame()
      except:
        pass
      max=min(X.shape[0], X.shape[1], y.shape[0],self.max_lv)
      hist_score=-np.inf
      flag=False
      for i in range(1,max+1):
        if flag==False:
          model=SIMPLS(n_components=i)
          model.fit(X,y)
          cv_score=cross_validate(model, X, y, cv=self.cv,scoring=self.scoring)['test_score'].sum()
          if cv_score>hist_score+self.epsilon:
            self.model=model
            self.__name__='PLS (n = '+str(i)+')'
            hist_score=cv_score
          else:
            flag=True
        self.n_components=self.model.n_components
        self.x_weights_=self.model.x_weights_
        self.y_weights_=self.model.y_weights_
        self.x_loadings_=self.model.x_loadings_
        self.y_loadings_=self.model.y_loadings_
        self.x_scores_=self.model.x_scores_
        self.y_scores_=self.model.y_scores_
        #self.xfrac_var_= np.var(self.x_scores_, axis = 0)/np.sum(np.var(X, axis = 0))
        #self.yfrac_var_=np.var(self.y_scores_, axis = 0)/np.sum(np.var(y, axis = 0))
        #self.xfrac_var_=np.trace(self.x_loadings_.T@self.x_loadings_)/(X.shape[0])
        #self.yfrac_var_=np.trace(self.y_loadings_.T@self.y_loadings_)/(X.shape[0])
        self.yfrac_var_=self.model.yfrac_var_
        self.xfrac_var_=self.model.xfrac_var_
        self.coef_=self.model.coef_
        self.B0=self.model.B0
        self.P2=1-np.sum((y-X@self.coef_)**2)
      return self

class InnerScaler(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='InnerScaler'
      self.var=None
    def fit(self,X,var):
      try:
        var=np.asarray(var)
      except:
        pass
      self.var=var
    def transform(self,X):
      try:
        X=np.asarray(X)
      except:
        pass
      try:
        size=len(self.var)
      except:
        size=1
      #s=np.ones((1,size))
      s=np.sqrt(self.var)
      
      s=s/np.sqrt(np.sum((X@s.T)**2,axis=0))
      square=np.zeros((len(s),len(s)))
      np.fill_diagonal(square,s)
      #print('xs',X@square)
      return X@square
    def fit_transform(self,X,var):
      self.fit(X,var)
      return self.transform(X)
      
class InnerScaler1(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='InnerScaler'
      self.var=None
      self.norm1=Normalizer()
      self.norm2=Normalizer()
    def fit(self,X,var):
      self.var=var
    def transform(self,X):
      try:
        X=np.asarray(X)
      except:
        pass
      try:
        X=self.norm1.fit_transform(X)
      except:
        X=self.norm1.fit_transform(X.reshape(-1,1))
      X=X*self.var
      shape=X.shape
      X=self.norm2.fit_transform(X.reshape(-1,1)).reshape(shape)
      return X
    def fit_transform(self,X,var):
      self.fit(X,var)
      return self.transform(X)
      
df=pd.read_csv(r'C:\\Users\\User\\Desktop\\processPLS\\ValdeLoirData.csv')
df=df.drop(columns=df.columns[0])
smell_at_rest=df.iloc[:,:5]
view=df.iloc[:,5:8]
smell_after_shaking=df.iloc[:,8:18]
tasting=df.iloc[:,18:27]
global_quality=df.iloc[:,27]

simpls=SIMPLS(3)
simpls.fit(df.iloc[:,:27],global_quality)


#i="1"+1
    
X={
'Smell at Rest':smell_at_rest,
"View":view,
"Smell after Shaking":smell_after_shaking,
"Tasting":tasting,
}

Y={"Global Quality":global_quality}

data={}
data.update(X)
data.update(Y)
outer_pls={}
outer_score={}
data11=r'C:\Users\User\Desktop\processPLS\Data.csv'

df = pd.DataFrame(
[
[0,0,0,0,0], 
[1,0,0,0,0],
[1,1,0,0,0],
[1,1,1,0,0],
[1,1,1,1,0],
],
index=list(data.keys()),
columns=list(data.keys())
)

df=df.T

G = nx.from_pandas_adjacency(df,create_using=nx.DiGraph)
YY=pd.DataFrame(list(Y.values())[0])
'''
for node_name in G:
    if node_name not in list(Y.keys()):
        simpls=PartialLeastSquaresCV()
        pipeline=make_pipeline(StandardScaler(),simpls).fit(X[node_name],YY)
        sx=np.ones_like(simpls.xfrac_var_)
        sx=sx*np.sqrt(sx*simpls.xfrac_var_)
        sx=np.divide(sx,np.sqrt(np.sum((simpls.x_scores_*sx.T)**2)))
        
        sy=np.ones_like(simpls.yfrac_var_)
        sy=sy*np.sqrt(sy*simpls.yfrac_var_)
        sy=np.divide(sy,np.sqrt(np.sum((simpls.y_scores_*sy.T)**2)))
        J=np.sum((simpls.y_scores_*sy)**2,axis=0)
        #print(np.sum((normalize((normalize(simpls.x_scores_,axis=0)*np.sqrt(simpls.frac_var_)), axis=0)*(simpls.x_scores_))**2,axis=0))
        #sy=s/np.sqrt(np.sum((simpls.y_scores_*s.T)**2))
        #print(simpls.x_scores_*np.diag(sx))
        G.add_node(node_name, outer_model=pipeline, x_scores=simpls.x_scores_*sx, y_scores=simpls.y_scores_*sy)
'''

######## OUTER MODEL###########

###Preprocessing the x data using standard scaler and place them in the correct nodes###
## Preprocess and put X in normal nodes ###
for node in G: #loop over all nodes in graph
    #only choose nodes which are not the 'end node'
    if node not in list(Y.keys()):
        outer_scaler_x=make_pipeline(StandardScaler())#define autoscaler 
        outer_x=outer_scaler_x.fit_transform(X[node]) #autoscale X data
        outer_x=np.divide(outer_x,np.sqrt(np.mean(outer_x**2)*outer_x.shape[1])) #sum of squares of 1
        G.add_node(node, outer_x= outer_x,outer_scaler=outer_scaler_x)

## Concatenate X, preprocess Y and put them in end nodes ###
#for 'end nodes' concatenate the preprocessed data of the predecessor nodes as input x
for node in list(Y.keys()):
    outer_x=np.array([]) #define array to store x #CHECK###
    #find predecessor of chosen end node
    for pre_nodes in list(G.predecessors(node)):
        f=G.nodes[pre_nodes]["outer_x"] #temporary variable for chosen pre node's outer X
        if outer_x.size==0: #if outer_x array is empty then just replace it with f
            outer_x=f
        else:    #otherwise concatenate them together
            outer_x=np.concatenate((outer_x,f),axis=1)        
    outer_scaler_y=make_pipeline(StandardScaler()) #define autoscaler 
    try:
        outer_y=outer_scaler_y.fit_transform(Y[node]) #autoscale Y data 
    except:
        outer_y=outer_scaler_y.fit_transform(Y[node].to_frame()) #reshape data if the input shape was somehow problematic 
    outer_y=np.divide(outer_y,np.sqrt(np.mean(outer_y**2)*outer_y.shape[1]))# sum of squares of 1
    G.add_node(node, outer_x= outer_x, outer_y=outer_y,outer_scaler=outer_scaler_y) #put the outer X and y into the end node


## Put the outer y into the normal nodes ##
for node in G:
    outer_y=np.array([]) #define array to store y
    if node not in list(Y.keys()):     #only choose nodes which are not the 'end node'
        for post_nodes in list(G.successors(node)): #loop through all normal node
            if post_nodes not in list(Y.keys()): #if the post node is a normal node
                f=G.nodes[post_nodes]["outer_x"] #map to the x of the post node
            else: #if the post node is an end node
                f=G.nodes[post_nodes]["outer_y"] #map to the y of the post node #### CHECK###########
            if outer_y.size==0: #if outer_y array is empty then just replace it with f
                outer_y=f
            else:    #otherwise concatenate them together
                outer_y=np.concatenate((outer_y,f),axis=1)     
        G.add_node(node,outer_y=outer_y)


## Make Outer Models ##
for node in G:
    simpls=PartialLeastSquaresCV(max_lv=30) #define simpls model
    simpls.fit(G.nodes[node]["outer_x"],G.nodes[node]["outer_y"]) #fit simpls model
    print(node," outer n_components")
    print(simpls.n_components)
    ##scale scores and weights for x
    inner_scaler_x=InnerScaler()
    outer_x_scores=inner_scaler_x.fit_transform(simpls.x_scores_,simpls.xfrac_var_)
    outer_x_weights=inner_scaler_x.transform(simpls.x_weights_)
    #print('scaled',outer_x_scores) #this is ok for 1 LV
    ##scale scores and weights for y
    inner_scaler_y=InnerScaler()
    outer_y_scores=inner_scaler_y.fit_transform(simpls.y_scores_,simpls.yfrac_var_)
    outer_y_weights=inner_scaler_y.transform(simpls.y_weights_) #CHECK THIS IS IT WEIGHTS OR LOADINGS?
    #print(outer_y_scores)
    G.add_node(node,outer_model=simpls,inner_scaler_x=inner_scaler_x,inner_scaler_y=inner_scaler_y,outer_x_scores=outer_x_scores,outer_x_weights=outer_x_weights,outer_y_scores=outer_y_scores,outer_y_weights=outer_y_weights) #put models in graph
    if node not in list(Y.keys()):     #only choose nodes which are not the 'end node'
        G.add_node(node,exp_var=simpls.xfrac_var_,R2m=np.trace(simpls.x_loadings_.T*simpls.x_loadings_.T)/(simpls.x_scores_.shape[0]-1))
    else:
        G.add_node(node,exp_var=simpls.yfrac_var_,R2m=np.trace(simpls.y_loadings_.T*simpls.y_loadings_.T)/(simpls.y_scores_.shape[0]-1))
    
    
######## INNER MODEL###########


##Make the inner models on edges##

for node in G: #loop through chosen nodes
    inner_x=np.array([]) #define array to store inner x
    inner_x_size=[] #define list to store inner x size accumulatively
    pred_list=list(G.predecessors(node)) #set the pred list for predecessor nodes
    if pred_list: #if predecessor is not empty
        for pre_node in pred_list: #check the successor nodes
            f=G.nodes[pre_node]["outer_x_scores"] #get the x_scores of successor
            inner_x_size.append(f.shape[1])
            if inner_x.size==0:
                inner_x=f #replace temp variable if empty
            else:
                inner_x=np.concatenate((inner_x,f),axis=1)  #otherwise combine the x scores 
        if node not in list(Y.keys()): #check if chosen node is an 'end node'
            inner_y= G.nodes[node]["outer_x_scores"]  #if no, then use x scores as response
        else:
            inner_y=G.nodes[node]["outer_y_scores"] #if yes, then use y scores as response 
        inner_simpls=PartialLeastSquaresCV()    # define inner model
        inner_simpls.fit(inner_x,inner_y)  #fit inner model
        print(node," inner n_components")
        print(inner_simpls.n_components)
        #print(inner_y)
        G.add_node(node,inner_model=inner_simpls, inner_model_x=inner_x, inner_model_y=inner_y) #place inner model on respective node
        #Put B coefficients on graph edges#
        inner_x_size.insert(0,-1) #fix index by prepending -1, which will be 0 later
        inner_x_size=np.cumsum(inner_x_size)+1 # add all by 1 for indexing
        B=inner_simpls.coef_ #get B coefficient of inner model
        X_LV=inner_x #get inner model prediction data,  X_LV 
        for i in range(len(pred_list)): #loop over predecessor node
            partialB=B[inner_x_size[i]:inner_x_size[i+1],:]  #get the correct partial B coefficients relative to predecessors
            partialX_LV=X_LV[:,inner_x_size[i]:inner_x_size[i+1]] #get the correct partial X_LV relative to predecessors
            Y_pred_part=partialX_LV@partialB #part prediction
            G.add_edge(pred_list[i],node,B=partialB,X_LV=partialX_LV, inner_y=inner_y, Y_pred_part=Y_pred_part) #put the correct B coeffficient,X_LV and inner y to edge



####PATH COEFFICIENT CALCULATION###
for node in G: #loop through nodes
    Y_pred=np.asarray([])
    Y_pred_sum=None
    pred_list=list(G.predecessors(node)) #get list of predecessor
    if pred_list: #if there is a predecessor
        for pre_node in pred_list: #loop through nodes
            B=G[pre_node][node]["B"]  #Get PLS Coefficient
            X_LV= G[pre_node][node]["X_LV"]  # get inner X data
            inner_y=G[pre_node][node]["inner_y"]  # get inner y data
            Y_pred_part= G[pre_node][node]["Y_pred_part"] #Get partial prediction from edge X*B partial
            #Add the partial prediction as total prediction
            if len(Y_pred)==0:
                Y_pred=Y_pred_part
            else:    
                Y_pred=Y_pred+Y_pred_part
            #E_pred_part=inner_y-Y_pred_part
            #Add sum of squares for total prediction
            if Y_pred_sum is None:
                Y_pred_sum=np.sum((Y_pred_part)**2)
            else:    
                Y_pred_sum=Y_pred_sum+np.sum((Y_pred_part)**2)
    
        #Calculate total prediction errors
        E_pred = inner_y-Y_pred

        for pre_node in pred_list: #loop through nodes
            inner_y=G[pre_node][node]["inner_y"]# get inner y data 
            #Calculate partial explained variance
            P2mz=(1-(np.sum(E_pred**2)/np.sum(inner_y**2)))*(np.sum(G[pre_node][node]["Y_pred_part"]**2)/Y_pred_sum)

            #Put this P2 path coefficient to edge of graph
            G.add_edge(pre_node,node, path_variances_explained=P2mz)
            
            print(pre_node, "-->", node)
            print("explained variances: ", P2mz)


####INDIRECT PATH EFFECTS####
for origin in G: #loop over origin
    for target in G: #loop over targets
        indirect=np.asarray([]) #R1 restart path indirect effects
        paths = nx.all_simple_paths(G,origin,target) #find all simple paths
        if paths: #if there is an object being returned
            for path in map(nx.utils.pairwise, paths): #loop through the pairwise steps
                pathlist=list(path) #define a path list
                if len(pathlist)>1: #if path length is larger than 1, then it is not a direct path but an indirect path
                    path_effects_indirect=np.asarray([]) #R2 Restart Path Effects
                    for tup in pathlist:  #look into tuple steps of the path
                        if path_effects_indirect.size==0:
                            path_effects_indirect=G[tup[0]][tup[1]]["B"]  #If path is new, take B coefficient initially
                        else:    
                            path_effects_indirect=path_effects_indirect@(G[tup[0]][tup[1]]["B"]) #Multiply B coefficient across path
                    if indirect.size==0:
                        indirect=path_effects_indirect  #R2 take path effects if indirect variable is not defined
                    else:    
                        indirect=indirect+path_effects_indirect #R2 Add indirect path effects of different paths with same node
        if len(indirect)!=0: #if there is an indirect effect
            G.add_edge(origin,target,indirect_effect=indirect) #add the indirect effect matrix in edge                     
        elif G.has_edge(origin,target):  #else if there is no indirect effect, but a connection with direct effect
            G.add_edge(origin,target,indirect_effect=0) #put indirect effect of 0

#### DIRECT EFFECTS AND FULL EFFECTS####

for origin, target in G.edges():
    G.add_edge(origin,target,direct_effect=G[origin][target]["B"]) #Put the direct effect as B coefficient
    G.add_edge(origin,target,full_effect=G[origin][target]["B"]+G[origin][target]["indirect_effect"]) #Put the full effect as sum of direct and indirect effects
    print(origin,"-->", target)
    print("full effect")
    print(G[origin][target]['full_effect'])
            

label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
edges=G.edges()
weights = [G[u][v]['path_variances_explained'] for u,v in edges]
labels = nx.get_edge_attributes(G,'path_variances_explained')
labels={k:round(v,2) for k,v in labels.items()}
nx.draw(G, pos=nx.shell_layout(G), node_color='black',alpha=0.9,node_shape='o', edge_color='black', with_labels = True,node_size=3000,bbox=label_options,width=weights)
nx.draw_networkx_edge_labels(G, pos=nx.shell_layout(G),edge_labels=labels)
ax= plt.gca()
plt.axis('off')
ax.set_xlim([1.5*x for x in ax.get_xlim()])
ax.set_ylim([1.5*y for y in ax.get_ylim()])
ax.set_aspect('equal', adjustable='box')
#plt.tight_layout()
plt.rc('figure',figsize=(15,15))
plt.show()

'''
for node in G:
    try:
        print(node)
        print(G.nodes[node]["outer_x_scores"])
        #print(G.nodes[node]["exp_var"])
        #print(innerscaler().fit_transform(G.nodes[node]["outer_x_scores"],G.nodes[node]["exp_var"]))
    except:
        pass
 '''   
