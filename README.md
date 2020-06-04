# SLIM
Structural Landmarking and Interaction Modelling: on Resolution Dilemmas in Graph Classification

### step1 Run or put the decompressed data folder in the data set project which you want，

eg.
------
    unzip -d /SLIM-MUTAG data.zip

### step2 Then unzip the n_LA_NCI1.pkl file to the current folder. If not, skip to the next step. 

### eg.in NCI1

    unzip 1order_LA_NCI1.zip
### step3 Then unzip the adj.pkl file to the current folder. If not, skip to the next step. 

### eg.in Mutag

    unzip adj_train3021.zip
### step4 
    sh slim.sh


![](https://github.com/Avigdor1231/SLIM/blob/master/SLIM-MUTAG/lib/test.jpg)
Organization of the code
------
* util.py (for data loading and basic data organization operators )
* main.py (for containing model, training and test code)
* graphVec.py (for using spatial content information to build features )
* Clustering.py (for clustering using DEC )
* predict.py (for fc layer and prediction results )
* slim.sh (for setting parameters and starting the entire project )
* n_LA_xxx.pkl(for saving the data of the nth node) 


