import numpy as np
def gini_node(y):
    y=np.array(y)
    values,counts=np.unique(y,return_counts=True)
    p=counts/len(y)
    return 1-np.matmul(p,p)

def gini_impurity(y_left, y_right):
    nl=len(y_left)
    nr=len(y_right)
    N=nl+nr
    tl=gini_node(y_left)
    tr=gini_node(y_right)
    if(N!=0):
        return (nl/N)*tl+(nr/N)*tr
    else:
        return 0.0
    