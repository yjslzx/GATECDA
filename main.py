import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import classifiers
import gate_feature
from data_generate import *
from data_preprocess import aupr, single_generate_graph_adj_and_feature
from sim_processing import get_syn_sim, sim_thresholding

# get sim matrix and association
seq_sim_matrix = pd.read_csv("mydata/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
str_sim_matrix = pd.read_csv("mydata/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
association = pd.read_csv("mydata/association.csv", index_col=0).to_numpy()

# super_parameter
n_splits = 5
classifier_epochs = 50

c_threshold = [0.7]  # generate subgraph
d_threshold = [0.6]
epochs=[200]
fold = 0

with open('mydata/new_result.txt', 'a') as result:
    result.write(f'{n_splits} folds test start:')

for alli in range(1):
    for s in itertools.product(c_threshold,d_threshold,epochs):
        val_label_scores = dict()
        # start: get and split samples from association matrix
        samples = get_all_samples(association)
        # end: get and split samples from association matrix

        # start: model parameters
        sum_auc_score = 0
        sum_aupr_score = 0
        # end: model parameters

        # start: define K-fold sample divider
        kf = KFold(n_splits=n_splits, shuffle=True)
        # end: define K-fold sample divider

        # start: train and evaluation model by k-Fold cross validation
        for train_index, val_index in kf.split(samples):
            fold += 1
            # start: split train and validation dataset from X_train
            train_samples = samples[train_index, :]
            val_samples = samples[val_index, :]
            # end: split train and validation dataset from X_train

            # start: feature extraction

            # delete test dataset association
            new_association = association.copy()
            for i in val_samples:
                new_association[i[0], i[1]] = 0

            # sim fusion
            c_sim, d_sim = get_syn_sim(new_association, seq_sim_matrix, str_sim_matrix, mode=1)

            # GATE
            print(s[0],s[1])
            c_network = sim_thresholding(c_sim, s[0])
            d_network = sim_thresholding(d_sim, s[1])
            c_adj, c_features = single_generate_graph_adj_and_feature(c_network, new_association)
            d_adj, d_features = single_generate_graph_adj_and_feature(d_network, new_association.T)
            c_embeddings = gate_feature.get_gate_feature(c_adj, c_features,s[2], 1)
            d_embeddings = gate_feature.get_gate_feature(d_adj, d_features,s[2], 1)

            # start: generate feature of train and validation dataset
            features = [c_embeddings, d_embeddings]
            train_feature, train_label = generate_f(train_samples, features)
            val_feature, val_label = generate_f(val_samples, features)
            # end: generate feature of train and validation dataset

            # start: build model
            model = classifiers.get_dnn()
            # end: build model

            # tensorflow fit
            history = model.fit(train_feature, train_label, validation_data=(val_feature, val_label), batch_size=64,
                                epochs=classifier_epochs, verbose=0)

            val_score = model.predict(val_feature)[:, 0]
            val_data = np.vstack((val_label, val_score)).tolist()
            val_label_scores['ROC fold ' + str(fold)] = val_data
            auc_score = roc_auc_score(val_label, val_score)
            aupr_score = aupr(val_label, val_score)
            print(f" auc: {auc_score}")
            sum_aupr_score += aupr_score
            sum_auc_score += auc_score
        print(f"lambda<{alli}>,cth<{s[0]}>,dth<{s[1]}>,{n_splits}-cv: auc {sum_auc_score / n_splits}, aupr {sum_aupr_score / n_splits}")
        # np.save(f'../methods_comparison_project/data/GATECDA/SelectParameter/GATECDA_{s[2]}_epochs_{n_splits}_cv_val_data.npy', val_label_scores)
        # end: train and evaluation model by k-Fold cross validation
