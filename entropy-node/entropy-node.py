import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y=np.array(y)
    values,counts=np.unique(y,return_counts=True)
    p=counts/len(y)
    log_p=np.log2(p)
    return -1*np.matmul(p,log_p)
     
        