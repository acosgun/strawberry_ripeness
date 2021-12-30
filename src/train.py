import os, sys
import numpy as np
import pickle

def main():

    from numpy import genfromtxt
    raw_data = genfromtxt('../data/data.csv', delimiter =',')
    raw_data_with_strings = genfromtxt('../data/data.csv', delimiter =',', dtype=None)
    
    X = []
    y = []

    for row in raw_data_with_strings:
        label_str = row[0].decode("utf-8")        
        #print(label_str)
        if label_str[:2] == "sr": # sr - red shade strawberry (ripe)
            y = np.append(y,0)
        elif  label_str[:2] == "sg": #sg - it means green shade strawberry (unripe)
            y = np.append(y,1)
        elif  label_str[:2] == "sp": #sp - plant plucked strawberry (overripe)
            y = np.append(y,2)
        else:
            #print("invalid label: " + label_str)
            pass
    #print(y)

    water_content_vals = []

    for row in raw_data:
        #print(len(row))
        #new_row = row[1:2]
        #print(row.type)
        wc_row = row[-5]
        
        water_content_vals = water_content_vals + [wc_row]

        row = row[1:]
        row = row[:-5]
        #X = np.row_stack((X, row))
        X = X + [row]
  
    water_content_vals = water_content_vals[1:]
  
    X = np.asarray(X)
    y = np.asarray(y)

    freq_labels = X[0,:] #from 350 to 2300
    X = X[1:,:] #remove first row which includes frequency labels

    index674 = 324
    index698 = 348

    #print(str(freq_labels[index674]))
    #print(str(freq_labels[index698]))

    # X is 1 numbers: 674 or 698
    # X = X[:,index674]
    # X = np.reshape(X,(-1,1))

    # X is 2 numbers: 674 or 698
    # X = X[:,[index674,index698]]

    # X is 1 number: NDSWI or SFWC
    # X_temp = []
    # for row in X:
    #     val674 = row[index674]
    #     val698 = row[index698]
    #     NDSWI = (val698-val674)/(val698+val674)
    #     import math
    #     SFWC = 0.038*math.log(NDSWI)+0.98
    #     X_temp = X_temp + [NDSWI]
    # X_temp = np.asarray(X_temp)
    # X_temp = np.reshape(X_temp,(-1,1))
    # X = X_temp

    #X is 1 number: Ground truth Water content
    # X = water_content_vals
    # X = np.reshape(X,(-1,1))

    # X is 2 numbers: WC and NDSWI
    # X_temp = []
    # for row in X:
    #     val674 = row[index674]
    #     val698 = row[index698]
    #     NDSWI = (val698-val674)/(val698+val674)
    #     import math
    #     SFWC = 0.038*math.log(NDSWI)+0.98
    #     X_temp = X_temp + [SFWC]
    # X_temp = [X_temp , water_content_vals]
    # X_temp = np.asarray(X_temp)
    # X_temp = X_temp.transpose() 
    # X = X_temp

    #X is 3 numbers: 674, 698 and WC
    # X = X[:,[index674,index698]]
    # water_content_vals = np.asarray(water_content_vals)
    # water_content_vals = np.reshape(water_content_vals,(-1,1))
    # X=np.hstack((X , water_content_vals))
    
    
    # X is all spectrum + water content
    water_content_vals = np.asarray(water_content_vals)
    water_content_vals = np.reshape(water_content_vals,(-1,1))
    X=np.hstack((X , water_content_vals))


    #print(X.shape)
    #exit(1)

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats = 25) #5, 25


    accuracies=[]
    class_0_correct = 0
    class_0_incorrect = 0
    class_1_correct = 0
    class_1_incorrect = 0
    class_2_correct = 0
    class_2_incorrect = 0
    
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import svm
        from sklearn.neural_network import MLPClassifier

        #Decision Tree
        #classifier = DecisionTreeClassifier()
        
        #SVM
        #classifier = svm.SVC(kernel='rbf')

        #MLP
        classifier = MLPClassifier(hidden_layer_sizes=(80,40,20,10), max_iter=1000, early_stopping = False, activation = 'relu', batch_size = 50)

        classifier.fit(X_train, y_train)
        pred_classes = classifier.predict(X_test)
    
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(pred_classes, y_test)
        accuracies = np.append(accuracies,accuracy)

        for index in range(len(y_test)):
            
            label = y_test[index]
            pred = pred_classes[index]
            
            if label == 0:
                if pred == label:
                    class_0_correct+=1
                else:
                    class_0_incorrect+=1
            elif label == 1:
                if pred == label:
                    class_1_correct+=1
                else:
                    class_1_incorrect+=1
            elif label == 2:
                if pred == label:
                    class_2_correct+=1
                else:
                    class_2_incorrect+=1

 
        print("accuracy: " + str(accuracy))
        #print(conf_matrix)
        #print ("False pos: " + str(false_positives) + " neg: " + str(false_negatives))

        #break
    #print(np.mean(accuracies))
    print("class_0_correct: " + str(class_0_correct))
    print("class_0_incorrect: " + str(class_0_incorrect))
    print("class_1_correct: " + str(class_1_correct))
    print("class_1_incorrect: " + str(class_1_incorrect))
    print("class_2_correct: " + str(class_2_correct))
    print("class_2_incorrect: " + str(class_2_incorrect))
    #print(accuracies)
    print("---")
    print("Overall: " + str(np.mean(accuracies)))
    print("Class 0 accuracy: " + str(class_0_correct/(class_0_correct+class_0_incorrect)))
    print("Class 1 accuracy: " + str(class_1_correct/(class_1_correct+class_1_incorrect)))
    print("Class 2 accuracy: " + str(class_2_correct/(class_2_correct+class_2_incorrect)))

#def train_decision_tree()

if __name__ == "__main__":
    main()