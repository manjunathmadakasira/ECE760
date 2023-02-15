import numpy as np
import pandas as pd
import math
import argparse
import matplotlib.pylab as plt

class Node:
    num_nodes = 0
    def __init__(self, feature=None, threshold=0, left=None, right=None, leaf=True,igr=0):
        self.leaf = leaf
        self.prediction = None
        self.feature = feature
        self.threshold = threshold
        self.left = left    
        self.right = right
        self.num_child_nodes = 0
        self.num_child_nodes_left = 0
        self.num_child_nodes_right = 0
        self.igr = igr
        if (leaf==False):
            type(self).num_nodes = type(self).num_nodes + 1

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
    give_info_gain = give_entropy(df) - give_conditional_entropy(df,feature,threshold)
    df_then = df[df[feature]>=threshold]
    df_else = df[df[feature]<threshold]
    p_then = float(df_then.shape[0]/df.shape[0])
    p_else = float(df_else.shape[0]/df.shape[0])
    #print("The two probabilities used in entropy split comp:",p_then,p_else)
    entropy_split = -p_then*np.log2(p_then) - p_else*np.log2(p_else)
    if entropy_split==0:
        give_info_gain_ratio = float("-inf")
    else:
        give_info_gain_ratio = give_info_gain/entropy_split
    return(give_info_gain_ratio,give_info_gain)

def get_threshold(df,feature):
    feature_values = df[feature].tolist()
    max_igr = float("-inf")
    selected_threshold = 0
    for i in range(len(feature_values)):
        threshold = feature_values[i]
        (igr,ig) = give_info_gain_ratio(df,feature,threshold)
        #print(feature,">=",threshold,igr,ig)
        if (igr!=float("-inf")):
            if(igr>max_igr):
                max_igr = igr
                selected_threshold = threshold
    return (selected_threshold,max_igr)

def get_feature(df, feature_list):
    max_igr = float("-inf")
    selected_feature = None
    selected_threshold = 0
    for f in feature_list:
        (threshold,igr) = get_threshold(df,f)
        #print("Compare the IGR from the earlier computation and in line computation:",igr,give_info_gain_ratio(df,f,threshold))
        if (igr > max_igr):
            max_igr = igr
            selected_feature = f
            selected_threshold = threshold
    return(selected_feature,selected_threshold,max_igr)

#Entropy is zero for all candidate splits if the item values are all same for a given feature
def all_candidate_split_entropy(df,feature_list):
    same_entries = True
    for f in feature_list:
        values = df[f].tolist()
        if (len(set(values))==1 and same_entries==True):
            same_entries=True
        else:
            same_entries=False
    return(same_entries)

def build_decision_tree(df,feature_list):
    num_true = df[df['y']==1].shape[0]
    num_false = df[df['y']==0].shape[0]

    #break condition:
    #If the node is empty: df.shape[0]==0
    #If the gain ratio is zero for any candidate split, ergo, all the items have the same label(either 0 or 1)
    #If entropy is zero for all candidate splits=>value is same across all items for all features

    if (num_true==0 or num_false==0 or all_candidate_split_entropy(df,feature_list)==True):
        nd = Node(leaf=True)
        nd.num_child_nodes = num_true + num_false
        if (num_true>=num_false):
            nd.prediction = 1
        else:
            nd.prediction = 0
        return nd
    else:
        #create a branch after selecting feature and threshold
        (f,thresh,igr) = get_feature(df,feature_list)
        split_then_df = df[df[f]>=thresh]
        split_else_df = df[df[f]<thresh]
        nd = Node(feature=f, leaf=False, threshold=thresh,igr=igr)
        nd.num_child_nodes = df.shape[0]
        nd.num_child_nodes_left = split_then_df.shape[0]
        nd.num_child_nodes_right = split_else_df.shape[0]
        print("Node constructed, now building other child nodes")
        nd.left = build_decision_tree(split_then_df,feature_list)
        nd.right = build_decision_tree(split_else_df,feature_list)
        return(nd)

#predict the label for a single point
def predict_single_point(nd , row_df):
    if nd.leaf:
        return nd.prediction
    if row_df[nd.feature] >= nd.threshold:
        return predict_single_point(nd.left, row_df)
    elif row_df[nd.feature] < nd.threshold:
        return predict_single_point(nd.right, row_df)

def give_accuracy(tree,df_test):

    num_data = df_test.shape[0]
    num_correct = 0
    for index,row in df_test.iterrows():
        prediction = predict_single_point(tree, row)
        if prediction == row['y']:
            num_correct += 1
    err0r = 1 - float(num_correct/num_data)
    return err0r

def plot_decision_boundary(tree,db_mode):
    predictions = []
    x1 = list(np.arange(-1.5,1.51,0.01))*300
    x2_pre = list(np.arange(-1.5,1.51,0.01))
    x2 = [ele for ele in x2_pre for _ in range(300)]
    df_2d = pd.DataFrame({'x1':x1,'x2':x2})
    for index,row in df_2d.iterrows():
        prediction = predict_single_point(tree, row)
        predictions.append(prediction)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision tree boundaries")
    plt.scatter(x1,x2,c=predictions,s=4) 
    plt.savefig('decision_boundary_'+str(db_mode)+'.png') 

        
def show_tree(nd=None, level=0,right_br=False,right_f='x1',right_thresh=0):
    if (right_br==True):
        print("|---","|---"*(level-1),right_f,"<",right_thresh)
    if (nd.leaf==True):
        print("|---","|---"*level,"class:",nd.prediction,"num_items:",nd.num_child_nodes)
    else:
        print("|---","|---"*level,nd.feature,">=",nd.threshold,"IGR:",nd.igr)
        show_tree(nd.left,level+1)
        show_tree(nd.right,level+1,True,nd.feature,nd.threshold)
    
def load_data(filename):
    print("running loading")
    # read text file into pandas DataFrame
    df = pd.read_csv(filename, sep=" ", header=None)
    df = df.rename(columns={0: 'x1', 1: 'x2', 2: 'y'})
    print(df)
    return(df)

def prepare_sub_data(df):
    df = df.sample(frac=1) #shuffle
    df_train_set = df.head(8192) #first N entries
    df_test_set = df.tail(1808)
    print("size of training set=",df_train_set.shape,"test set=",df_test_set.shape)
    df_train_set_nested_32 = df_train_set.head(32)
    df_train_set_nested_128 = df_train_set.head(128)
    df_train_set_nested_512 = df_train_set.head(512)
    df_train_set_nested_2048 = df_train_set.head(2048)
    df_train_set_nested_8192 = df_train_set.head(8192)
    np.savetxt(r'D32.txt', df_train_set_nested_32.values, fmt='%d')
    np.savetxt(r'D128.txt', df_train_set_nested_128.values, fmt='%d')
    np.savetxt(r'D512.txt', df_train_set_nested_512.values, fmt='%d')
    np.savetxt(r'D2048.txt', df_train_set_nested_2048.values, fmt='%d')
    np.savetxt(r'D8192.txt', df_train_set_nested_8192.values, fmt='%d')
    np.savetxt(r'Dtest.txt', df_test_set.values, fmt='%d')
    return ([df_train_set_nested_32,df_train_set_nested_128,df_train_set_nested_512,df_train_set_nested_2048,df_train_set_nested_8192],df_test_set)

def parse_arguments():
    parser = argparse.ArgumentParser(description ='Search some files')
    parser.add_argument('--data_file', dest ='dfile',
                         default ='Druns.txt', help ='Data file with training/test points')
    parser.add_argument('-v', dest ='verbose',
                    action ='store_true', help ='verbose mode')
    #parser.add_argument('--dbig_mode', dest ='db_mode',type=int,
    #                     default =32, help ='Training data size mode for 2.7 question.',choices=[32,128,512,2048,8192])
    args = parser.parse_args()
    return args

def main():
    print("Running main")
    feature_list = ['x1','x2']

    args = parse_arguments()
    print("Data file is",args.dfile)
    df = load_data(args.dfile)
    if (args.dfile=="Dbig.txt"):
        (df_train_set,df_test_set) = prepare_sub_data(df)
        for idx,db_mode in enumerate ([32,128,512,2048,8192]):
            tree=None
            tree = build_decision_tree(df_train_set[idx],feature_list)
            accuracy = give_accuracy(tree,df_test_set)
            print("Accuracy is ",accuracy)
            print("Number of nodes is ",tree.num_nodes)
            plot_decision_boundary(tree,db_mode)
    else:
        tree = build_decision_tree(df,feature_list)
    #show_tree(tree)

if __name__ == "__main__":
    main()
