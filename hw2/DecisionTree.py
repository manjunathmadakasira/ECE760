import numpy as np
import pandas as pd
import math
import argparse

class Node:
    def __init__(self, feature=None, threshold=0, left=None, right=None, leaf=True):
        self.leaf = leaf
        self.prediction = None
        self.feature = feature
        self.threshold = threshold
        self.left = left    
        self.right = right
        self.num_child_nodes = 0
        self.num_child_nodes_left = 0
        self.num_child_nodes_right = 0

def give_entropy(df):
    #entropy calc
    true_df = df[df['y']==1]
    false_df = df[df['y']==0]
    if (df.shape[0]==0):
        p_true = 0
        p_false = 0
    else:
        p_true = float(true_df.shape[0]/df.shape[0])
        p_false = float(false_df.shape[0]/df.shape[0])
    if (p_true==0 or p_false==0):
        give_entropy = 0
    else:
        give_entropy = -p_true*np.log2(p_true) - p_false*np.log2(p_false)
    return give_entropy

def give_conditional_entropy(df, feature, threshold):
    #cond entropy calc
    df_then = df[df[feature]>=threshold]
    df_else = df[df[feature]<threshold]
    give_conditional_entropy = float(df_then.shape[0]/df.shape[0])*give_entropy(df_then) + float(df_else.shape[0]/df.shape[0])*give_entropy(df_else)
    return(give_conditional_entropy)

def give_split_entropy(df,feature,threshold):
    df_then = df[df[feature]>=threshold]
    df_else = df[df[feature]<threshold]
    p_then = float(df_then.shape[0]/df.shape[0])
    p_else = float(df_else.shape[0]/df.shape[0])
    give_split_entropy = -p_then*np.log2(p_then) - p_else*np.log2(p_else)
    return(give_split_entropy)

def give_info_gain_ratio(df, feature, threshold):
    #infogain ratio
    give_info_gain_ratio = give_entropy(df) - give_conditional_entropy(df,feature,threshold)
    df_then = df[df[feature]>=threshold]
    df_else = df[df[feature]<threshold]
    p_then = float(df_then.shape[0]/df.shape[0])
    p_else = float(df_else.shape[0]/df.shape[0])
    print("The two probabilities used in entropy split comp:",p_then,p_else)
    entropy_split = -p_then*np.log2(p_then) - p_else*np.log2(p_else)
    give_info_gain_ratio = give_info_gain_ratio/entropy_split
    return(give_info_gain_ratio)

def get_threshold(df,feature):
    feature_values = df[feature].tolist()
    max_igr = float("-inf")
    selected_threshold = 0
    for i in range(len(feature_values)):
        threshold = feature_values[i]
        igr = give_info_gain_ratio(df,feature,threshold)
        if (igr>max_igr):
            max_igr = igr
            selected_threshold = threshold
    return (selected_threshold,max_igr)

def get_feature(df, feature_list):
    max_igr = float("-inf")
    selected_feature = None
    selected_threshold = 0
    for f in feature_list:
        (threshold,igr) = get_threshold(df,f)
        print("Compare the IGR from the earlier computation and in line computation:",igr,give_info_gain_ratio(df,f,threshold))
        if (igr > max_igr):
            max_igr = igr
            selected_feature = f
            selected_threshold = threshold
    return(selected_feature,selected_threshold,max_igr)

def build_decision_tree(df,feature_list):
    num_true = df[df['y']==1].shape[0]
    num_false = df[df['y']==0].shape[0]

    #break condition
    if (num_true==0 or num_false==0):
        nd = Node()
        nd.leaf = True
        nd.num_child_nodes = num_true + num_false
        if (num_true>=num_false):
            nd.prediction = True
        else:
            nd.prediction = False
        return nd
    else:
        #create a branch after selecting feature and threshold
        (f,thresh,igr) = get_feature(df,feature_list)
        split_then_df = df[df[f]>=thresh]
        split_else_df = df[df[f]<thresh]
        nd = Node(feature=f, leaf=False, threshold=thresh)
        nd.num_child_nodes = df.shape[0]
        nd.num_child_nodes_left = split_then_df.shape[0]
        nd.num_child_nodes_right = split_else_df.shape[0]

        nd.left = build_decision_tree(split_then_df,feature_list)
        nd.right = build_decision_tree(split_else_df,feature_list)
        return(nd)
        
def show_tree(nd=None, level=0,right_br=False,right_f='x1',right_thresh=0):
    if (right_br==True):
        print("|---","|---"*(level-1),right_f,"<",right_thresh)
    if (nd.leaf==True):
        print("|---","|---"*level,"class:",nd.prediction,"num_items:",nd.num_child_nodes)
    else:
        print("|---","|---"*level,nd.feature,">=",nd.threshold)
        show_tree(nd.left,level+1)
        show_tree(nd.right,level+1,True,nd.feature,nd.threshold)
    

def load_data(filename):
    print("running loading")
    # read text file into pandas DataFrame
    df = pd.read_csv(filename, sep=" ", header=None)
    df = df.rename(columns={0: 'x1', 1: 'x2', 2: 'y'})
    print(df)
    return(df)

def parse_arguments():
    parser = argparse.ArgumentParser(description ='Search some files')
    parser.add_argument('--data_file', dest ='dfile',
                         default ='Druns.txt', help ='Data file with training/test points')
    parser.add_argument('-v', dest ='verbose',
                    action ='store_true', help ='verbose mode')
    args = parser.parse_args()
    return args

def main():
    print("Running main")
    args = parse_arguments()
    print("Data file is",args.dfile)
    df = load_data(args.dfile)
    feature_list = ['x1','x2']
    tree = build_decision_tree(df,feature_list)
    show_tree(tree)

if __name__ == "__main__":
    main()
