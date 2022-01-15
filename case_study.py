import pandas as pd

import classifiers
import gate_feature
from data_generate import *
from data_preprocess import single_generate_graph_adj_and_feature
from sim_processing import get_syn_sim, sim_thresholding, get_syn_sim_circ_drug


def get_all_pred_samples(conjunction):
    pred=[]
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
                pred.append([index, col, 1])
    pred=np.array(pred)
    return pred


# get sim matrix and association
seq_sim_matrix = pd.read_csv("mydata/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
str_sim_matrix = pd.read_csv("mydata/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
association = pd.read_csv("mydata/association.csv", index_col=0).to_numpy()

# super_parameter
latent_dims = [128]
n_splits = 10
classifier_epochs = 50
c_threshold = 0.7 # generate subgraph
d_threshold = 0.6  # generate subgraph

pred_score_matrix=np.zeros((271,218))
for index in range(10):
    print(f"{index}-th prediction start...")
    # start: get and split samples from association matrix
    samples = get_all_samples(association)
    # end: get and split samples from association matrix

    # start: feature extraction
    print("start: feature extraction")
    # delete test dataset association
    c_sim, d_sim = get_syn_sim(association, seq_sim_matrix, str_sim_matrix, mode=1)

    # GATE
    c_network = sim_thresholding(c_sim, c_threshold)
    d_network = sim_thresholding(d_sim, d_threshold)
    c_adj, c_features = single_generate_graph_adj_and_feature(c_network, association)
    d_adj, d_features = single_generate_graph_adj_and_feature(d_network, association.T)
    c_embeddings = gate_feature.get_gate_feature(c_adj, c_features, 200, 1)
    d_embeddings = gate_feature.get_gate_feature(d_adj, d_features, 200, 1)
    # end: feature extraction
    features = [c_embeddings, d_embeddings]
    feature, label = generate_f(samples, features)
    # end: generate feature of train and validation dataset

    # start: build model
    model = classifiers.get_dnn()
    # end: build model

    # tensorflow fit
    print("start model fit")
    history = model.fit(feature, label, batch_size=64,
                        epochs=classifier_epochs, verbose=0)

    print("start prediction")
    pred_samples = get_all_pred_samples(association)
    pred_feature, pred_label = generate_f(pred_samples, features)
    pred_score = model.predict(pred_feature)[:, 0]
    for i in range(len(pred_score)):
        pred_score_matrix[pred_samples[i,0], pred_samples[i,1]]+=pred_score[i]

pred_score=pd.DataFrame(pred_score_matrix/10)
pred_score.to_csv("mydata/pred_results.csv")
pass