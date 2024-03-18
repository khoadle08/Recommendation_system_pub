import pandas as pd
import numpy as np

def Pre_data(link:str):
    df=pd.read_csv(link)
    adjacency_list=df.values.tolist()
    size = max(max(adjacency_list)) + 1


    # Initialize an empty adjacency matrix
    adjacency = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if i==j:
                adjacency[i][j]=2
    # Populate the matrix for each edge
    for sink, source in adjacency_list:
        adjacency[source-1][sink-1]= 1
        adjacency[sink-1][source-1] = 1
    
    return np.array(adjacency)
