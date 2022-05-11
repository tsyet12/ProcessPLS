from sklearn.model_selection import KFold, RepeatedKFold
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, Normalizer
#from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import warnings
def warn(*args, **kwargs):
    pass


def svd_signstable(X):
    ###Based on Bro, R., Acar, E. and Kolda, T.G., 2008. Resolving the sign ambiguity in the singular value decomposition. Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), pp.135-140.
    try:
        X=np.asarray(X)
    except:
        pass    
    U, D, V= np.linalg.svd(X,full_matrices=False)
    V=V.T  #python V is transposed compared to matlab
    K=len(D)
    s_left=np.zeros((1,K))
    
    #step 1
    for k in range(K):
        select=np.setdiff1d(list(range(K)),k)
        DD=np.zeros((K-1,K-1))
        np.fill_diagonal(DD,D[select])
        Y=X-U[:,select]@DD@V[:,select].T
        
        s_left_parts= np.zeros((1,Y.shape[1]))
        
        for j in range(Y.shape[1]):
            temp_prod=(U[:,k].T)@(Y[:,j])
            s_left_parts[:,j]=np.sign(temp_prod)*(temp_prod**2)
        
        s_left[:,k]=np.sum(s_left_parts)
        
    #step 2
    s_right=np.zeros((1,K)) 
    for k in range(K):
        select=np.setdiff1d(list(range(K)),k)
        DD=np.zeros((K-1,K-1))
        np.fill_diagonal(DD,D[select])
        Y=X-U[:,select]@DD@V[:,select].T
        
        s_right_parts=np.zeros((1,Y.shape[0]))
        for i in range(Y.shape[0]):
            temp_prod= (V[:,k].T)@(Y[i,:].T)
            s_right_parts[:,i]=np.sign(temp_prod)*(temp_prod**2)
        s_right[:,k]=np.sum(s_right_parts)    

    #step 3
    for k in range(K):
        if (s_right[:,k]*s_left[:,k])<0:
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
      #Initialize vectors
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
      #X=X-np.mean(X,axis=0)
      y=y-np.mean(y,axis=0)
      S=X.T@y  #Covariance matrix

      for i in range(self.n_components): #loop through the latent variables      
        _,_,q=svd_signstable(S) #solve the sign stable svd
        
        q=q[:,0]
         
        r=S@q  #X block factor weight
        t=X@r #X block factor scores
        
        t=t-np.mean(t,axis=0)
        
        #normt=np.linalg.norm(t)
        normt=np.sqrt(t.T@t)
        
        
        t=np.divide(t,normt)
        r=np.divide(r,normt)
        
        p=X.T@t
        q=y.T@t
        
        self.x_loadings_[:,i]=p
        self.y_loadings_[:,i]=q

        
        # scores and weights
        self.x_scores_[:,i]=t
        self.y_scores_[:,i]=(y@q)
        self.x_weights_[:,i]=r

        #update orthonormal basis
        vi=np.copy(self.x_loadings_[:,i])

        
        if i>0:
            vi=vi-self.var_@(self.var_.T@p)
            self.y_scores_[:,i]=self.y_scores_[:,i]-self.x_scores_@(self.x_scores_.T@self.y_scores_[:,i]) #orthogonalize Y scores to preceeding X scores
        vi=np.divide(vi,np.sqrt(vi.T@vi))
        self.var_[:,i]=vi#save
        vi=vi.reshape(-1,1)
        S=S-vi@(vi.T@S)
        
     
      #calculate B regression vector      
      self.coef_=self.x_weights_@self.y_loadings_.T
      self.B0=np.mean(y,axis=0)-np.mean(X,axis=0)@self.coef_
      
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
  def __init__(self,cv=RepeatedKFold(n_splits=5,n_repeats=2,random_state=999),scoring='neg_mean_squared_error',epsilon=+0.0001,max_lv=30,forced_lv=None):
      self.__name__='PartialLeastSquaresCV'
      self.cv=cv
      self.model=None
      self.scoring=scoring
      self.epsilon=epsilon
      self.max_lv=max_lv
      self.forced_lv=forced_lv
      
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
      
      if self.forced_lv is None:
          for i in range(1,max+1):
            if flag==False:
              model=SIMPLS(n_components=i)#,scale=False)
              model.fit(X,y)
              cv_score=cross_validate(model, X, y, cv=self.cv,scoring=self.scoring)['test_score'].sum()
              if cv_score>hist_score+self.epsilon:
                self.model=model
                self.__name__='SIMPLS (n = '+str(i)+')'
                hist_score=cv_score
              else:
                flag=False
      else:
          model=SIMPLS(n_components=self.forced_lv) #,scale=False)
          model.fit(X,y)
          self.model=model
          self.__name__='SIMPLS (n = '+str(self.forced_lv)+')'
      self.n_components=self.model.n_components
      self.x_weights_=self.model.x_weights_
      self.y_weights_=self.model.y_weights_
      self.x_loadings_=self.model.x_loadings_
      self.y_loadings_=self.model.y_loadings_
      self.x_scores_=self.model.x_scores_
      self.y_scores_=self.model.y_scores_
      self.yfrac_var_=self.model.yfrac_var_
      self.xfrac_var_=self.model.xfrac_var_
      self.coef_=self.model.coef_
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
      size=self.var.shape[0]
      s=np.ones((1,size))
      
      s=s*np.sqrt(s*self.var)
      
      s=s/np.sqrt(np.sum((X@s.T)**2,axis=0))

      square=np.zeros((len(s[0]),len(s[0])))
      np.fill_diagonal(square,s)

      return X@square
    def fit_transform(self,X,var):
      self.fit(X,var)
      return self.transform(X)

class OuterScaler(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='OuterScaler'
      self.Xmean=None
      self.Xstd=None
      self.Xss=None
    def fit(self,X):
      try:
        X=np.asarray(X)
      except:
        pass
      self.Xmean=np.mean(X,axis=0)
      self.Xstd=np.std(X,axis=0)
      self.Xss=np.sqrt(np.mean(((X-self.Xmean)/self.Xstd)**2)*X.shape[1])
    def transform(self,X):
      try:
        X=np.asarray(X)
      except:
        pass
      X=(X-self.Xmean)/self.Xstd #Autoscale
      X=np.divide(X,self.Xss) #norm2 scale
      return X
    def fit_transform(self,X):
      self.fit(X)
      return self.transform(X)
    def inverse_transform(self,X):
      try:
        X=np.asarray(X)
      except:
        pass
      X=X*self.Xss
      X=X*self.Xstd+self.Xmean
      return X
class ProcessPLS(BaseEstimator):
  def __init__(self,cv=RepeatedKFold(n_splits=5,n_repeats=2,random_state=999),scoring='neg_mean_squared_error',max_lv=30,overwrite_lv=False,inner_forced_lv=None,outer_forced_lv=None,name=None):
      self.__name__='ProcessPLS'
      self.cv=cv
      self.name=name
      self.max_lv=max_lv
      self.scoring=scoring
      self.overwrite_lv=overwrite_lv
      self.inner_forced_lv=inner_forced_lv
      self.outer_forced_lv=outer_forced_lv
      self.G=None
      warnings.warn = warn
  def fit(self,X,y, matrix):
      ####Initialize####
      matrix=matrix.T
      G = nx.from_pandas_adjacency(matrix,create_using=nx.DiGraph) #Make the directed graph
      inner_forced_lv=self.inner_forced_lv
      outer_forced_lv=self.outer_forced_lv
      scoring=self.scoring
      max_lv=self.max_lv
      cv=self.cv
      overwrite_lv=self.overwrite_lv
      self.X=X
      self.Y=y
      Y=y
      ######## OUTER MODEL###########
      ###Preprocessing the x data using standard scaler and place them in the correct nodes###
      ## Preprocess and put X in normal nodes ###
      for node in G: #loop over all nodes in graph
          #only choose nodes which are not the 'end node'
          if node not in list(Y.keys()):
              #outer_scaler_x=make_pipeline(StandardScaler())#define autoscaler 
              outer_scaler_x=OuterScaler()
              outer_x=outer_scaler_x.fit_transform(X[node]) #autoscale and unit scale sum of squares  X data
              G.add_node(node, outer_x= outer_x,outer_scaler=outer_scaler_x)

      ## Concatenate X, preprocess Y and put them in end nodes ###
      #for 'end nodes' concatenate the preprocessed data of the predecessor nodes as input x
      for node in list(Y.keys()):
          outer_x=np.array([]) #define array to store x 
          #find predecessor of chosen end node
          for pre_nodes in list(G.predecessors(node)):
              f=G.nodes[pre_nodes]["outer_x"] #temporary variable for chosen pre node's outer X
              if outer_x.size==0: #if outer_x array is empty then just replace it with f
                  outer_x=f
              else:    #otherwise concatenate them together
                  outer_x=np.concatenate((outer_x,f),axis=1)        
          
          outer_scaler_y=OuterScaler()
          try:
              outer_y=outer_scaler_y.fit_transform(Y[node]) #autoscale Y data 
          except:
              outer_y=outer_scaler_y.fit_transform(Y[node].to_frame()) #reshape data if the input shape was somehow problematic 
          G.add_node(node, outer_x= outer_x, outer_y=outer_y,outer_scaler=outer_scaler_y) #put the outer X and y into the end node

      ## Put the outer y into the normal nodes ##
      for node in G:
          outer_y=np.array([]) #define array to store y
          if node not in list(Y.keys()):     #only choose nodes which are not the 'end node'
              for post_nodes in list(G.successors(node)): #loop through all normal node
                  if post_nodes not in list(Y.keys()): #if the post node is a normal node
                      f=G.nodes[post_nodes]["outer_x"] #map to the x of the post node
                  else: #if the post node is an end node
                      f=G.nodes[post_nodes]["outer_y"] #map to the y of the post node 
                  if outer_y.size==0: #if outer_y array is empty then just replace it with f
                      outer_y=f
                  else:    #otherwise concatenate them together
                      outer_y=np.concatenate((outer_y,f),axis=1)     
              G.add_node(node,outer_y=outer_y)

      ## Make Outer Models ##
      for node in G:
          simpls=PartialLeastSquaresCV(max_lv=max_lv,cv=cv,scoring=scoring) #define simpls model
          if overwrite_lv==True:
              if outer_forced_lv[node]!=None:  # replace simpls model if a forced lv is specified
                  simpls=PartialLeastSquaresCV(forced_lv=outer_forced_lv[node],cv=cv,scoring=scoring) 
          simpls.fit(G.nodes[node]["outer_x"],G.nodes[node]["outer_y"]) #fit simpls model
          ##scale scores and weights for x
          inner_scaler_x=InnerScaler()
          outer_x_scores=inner_scaler_x.fit_transform(simpls.x_scores_,simpls.xfrac_var_)
          outer_x_weights=inner_scaler_x.transform(simpls.x_weights_)
          ##scale scores and weights for y
          inner_scaler_y=InnerScaler()
          outer_y_scores=inner_scaler_y.fit_transform(simpls.y_scores_,simpls.yfrac_var_)
          outer_y_weights=inner_scaler_y.transform(simpls.y_weights_) #y weights are scaled
          G.add_node(node,outer_model=simpls,inner_scaler_x=inner_scaler_x,inner_scaler_y=inner_scaler_y,outer_x_scores=outer_x_scores,outer_x_weights=outer_x_weights,outer_y_scores=outer_y_scores,outer_y_weights=outer_y_weights) #put models in graph
          if node not in list(Y.keys()):     #only choose nodes which are not the 'end node'
              G.add_node(node,exp_var=simpls.xfrac_var_,R2m=np.trace(simpls.x_loadings_.T*simpls.x_loadings_.T)/(simpls.x_scores_.shape[0]-1))
          else:
              G.add_node(node,exp_var=simpls.yfrac_var_,R2m=np.trace(simpls.y_loadings_.T*simpls.y_loadings_.T)/(simpls.y_scores_.shape[0]-1))
          
      ######## INNER MODEL###########
      ##Make the inner models on target nodes##
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
              inner_simpls=PartialLeastSquaresCV(cv=cv,scoring=scoring)    # define inner model
              if overwrite_lv==True:
                  if inner_forced_lv[node]!=None:  # replace simpls model if a forced lv is specified ########## CHECK intialization
                      inner_simpls=PartialLeastSquaresCV(forced_lv=inner_forced_lv[node],cv=cv,scoring=scoring) 
              inner_simpls.fit(inner_x,inner_y)  #fit inner model
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
                          real_path=True #check if not skipped nodes, creating false path
                          for tup in pathlist:  #look into tuple steps of the path
                            if real_path:
                              try:
                                if path_effects_indirect.size==0:
                                    path_effects_indirect=G[tup[0]][tup[1]]["B"]  #If path is new, take B coefficient initially
                                    pass
                                else:    
                                    path_effects_indirect=path_effects_indirect@(G[tup[0]][tup[1]]["B"]) #Multiply B coefficient across path
                              except:
                                path_effects_indirect=0
                                real_path=False
                          if indirect.size==0:
                              indirect=path_effects_indirect  #R2 take path effects if indirect variable is not defined
                          else:    
                              indirect=indirect+path_effects_indirect #R2 Add indirect path effects of different paths with same node
              if G.has_edge(origin,target): #only record indirect effects when there is a direct effect
                if len(indirect)!=0: #if there is an indirect effect
                    G.add_edge(origin,target,indirect_effect=indirect) #add the indirect effect matrix in edge                     
                elif G.has_edge(origin,target):  #else if there is no indirect effect, but a connection with direct effect
                    G.add_edge(origin,target,indirect_effect=0) #put indirect effect of 0

      #### DIRECT EFFECTS AND FULL EFFECTS####
      for origin, target in G.edges():
        try:
          G.add_edge(origin,target,direct_effect=G[origin][target]["B"]) #Put the direct effect as B coefficient
          G.add_edge(origin,target,full_effect=G[origin][target]["B"]+G[origin][target]["indirect_effect"]) #Put the full effect as sum of direct and indirect effects
          print(origin,"-->",target," LV on LV full effect :")
          print(G[origin][target]['full_effect']) #LV on LV total
        except:
          pass
                  
      #### EFFECTS OF MV ON LV #######

      for node in G:
          if node not in list(Y.keys()):
              var_cont=np.sum(G.nodes[node]["outer_model"].x_loadings_*G.nodes[node]["outer_model"].xfrac_var_,1)/np.sum(G.nodes[node]["outer_model"].xfrac_var_)
          else:
              var_cont=np.sum(G.nodes[node]["outer_model"].y_loadings_*G.nodes[node]["outer_model"].yfrac_var_,1)/np.sum(G.nodes[node]["outer_model"].yfrac_var_)
          G.add_node(node, variable_contributions=var_cont)
      #### CALCULATE EFFECTS OF MVS ON OTHER LVS and blocks (INNER EFFECTS) #####
      for origin, target in G.edges():
        try:
          G.add_edge(origin,target, MV_on_other_LV=G.nodes[origin]["outer_model"].x_weights_@G[origin][target]['full_effect']) #MVS ON OTHER LVS
          G.add_edge(origin,target, MV_on_other_blocks=np.sum(G[origin][target]['MV_on_other_LV'],axis=1))####MV on other blocks####
          print(origin, '-->', target,' MV on LV Effects :', )
          print(G[origin][target]['MV_on_other_LV'])
          print(origin, '-->', target,' MV on other blocks Effects :', )
          print(G[origin][target]['MV_on_other_blocks'])
        except:
          pass
      self.G=G #update graph
  
  def predict(self, X,y=None):
    #initialize###
    Y=self.Y
    G=self.G
    for node in G: #loop over all nodes in graph
          #only choose nodes which are not the 'end node'
          if node not in list(Y.keys()):
              outer_scaler_x=G.nodes[node]["outer_scaler"]
              outer_x=outer_scaler_x.transform(X[node]) #autoscale and unit scale sum of squares  X data
              G.nodes[node]['outer_x']= outer_x 
    

    ## Concatenate X, preprocess Y and put them in end nodes ###
    #for 'end nodes' concatenate the preprocessed data of the predecessor nodes as input x
    for node in list(Y.keys()):
        outer_x=np.array([]) #define array to store x 
        #find predecessor of chosen end node
        for pre_nodes in list(G.predecessors(node)):
            f=G.nodes[node]['outer_x'] #temporary variable for chosen pre node's outer X
            if outer_x.size==0: #if outer_x array is empty then just replace it with f
                outer_x=f
            else:    #otherwise concatenate them together
                outer_x=np.concatenate((outer_x,f),axis=1)        
      
        simpls=G.nodes[node]["outer_model"]
        y_pred=simpls.predict(G.nodes[node]["outer_x"]) #predict simpls model
        outer_scaler_y=G.nodes[node]["outer_scaler"]
    return outer_scaler_y.inverse_transform(y_pred)
    
  def plot(self,figsize=(6,6)):
    plt.rc('figure',figsize=figsize)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.5}
    edges=self.G.edges()
    weights = [self.G[u][v]['path_variances_explained'] for u,v in edges]
    labels = nx.get_edge_attributes(self.G,'path_variances_explained')
    labels={k:round(v,2) for k,v in labels.items()}

    nx.draw(self.G, pos=nx.shell_layout(self.G,rotate=math.pi/len(self.G)*(len(self.G)/2)), node_color='black',alpha=0.9,node_shape='o', edge_color='black', with_labels = True,node_size=3000,bbox=label_options,width=weights,font_size=12)
    nx.draw_networkx_edge_labels(self.G, pos=nx.shell_layout(self.G,rotate=math.pi/len(self.G)*(len(self.G)/2)),edge_labels=labels)
    ax= plt.gca()
    plt.axis('off')
    ax.set_xlim([1.5*x for x in ax.get_xlim()])
    ax.set_ylim([1.5*y for y in ax.get_ylim()])
    ax.set_aspect('equal', adjustable='box')
    if self.name is not None:
      ax.set_title(self.name,y=0.83)
    plt.show()





if __name__=="__main__":      
  df=pd.read_csv(r'C:\Users\User\Desktop\processPLS\processPLS\data\ValdeLoirData.csv')
  df=df.drop(columns=df.columns[0])
  smell_at_rest=df.iloc[:,:5]
  view=df.iloc[:,5:8]
  smell_after_shaking=df.iloc[:,8:18]
  tasting=df.iloc[:,18:27]
  global_quality=df.iloc[:,27]
  
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
  
  #### MATLAB 1 LV CONSISTENCY TEST####
  Model1LV=ProcessPLS(cv=RepeatedKFold(n_splits=5,n_repeats=1,random_state=999), max_lv=1,name="Process PLS with 1 Outer LV ")
  Model1LV.fit(X,Y,matrix)
  print(Model1LV.predict(X))
  print(global_quality)
  Model1LV.plot()
  
  #################
  
  #### MATLAB WLV CONSISTENCY TEST###
  outer_forced_lv={
  'Smell at Rest':3,
  "View":3,
  "Smell after Shaking":2,
  "Tasting":5,
  "Global Quality":3
  }

  inner_forced_lv={
  'Smell at Rest':None,
  "View":3,
  "Smell after Shaking":6,
  "Tasting":8,
  "Global Quality":13
  }
  
  ModelWLV=ProcessPLS(cv=RepeatedKFold(n_splits=5,n_repeats=1,random_state=999), max_lv=np.inf,overwrite_lv=True,inner_forced_lv=inner_forced_lv,outer_forced_lv=outer_forced_lv, name="Process PLS with W Outer LV")
  ModelWLV.fit(X,Y,matrix)
  ModelWLV.plot()


  
  #######
  
  