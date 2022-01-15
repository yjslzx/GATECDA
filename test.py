import pandas as pd
from data_preprocess import generate_graph_adj_and_feature, prepare_graph_data, parse_args
from data_generate import *
from sim_processing import get_syn_sim, sim_thresholding
from gate_trainer import GATETrainer
import tensorflow._api.v2.compat.v1 as tf


seq_sim_matrix = pd.read_csv("mydata/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
str_sim_matrix = pd.read_csv("mydata/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
association = pd.read_csv("mydata/association.csv", index_col=0).to_numpy()

# c_sim, d_sim = get_syn_sim(association, seq_sim_matrix, str_sim_matrix)
#
# threshold = 0.5  # generate subgraph
# c_args = parse_args()
# d_args = parse_args()
# c_network = sim_thresholding(c_sim, threshold)
# d_network = sim_thresholding(d_sim, threshold)
# c_adj, d_adj, c_features, d_features = generate_graph_adj_and_feature(c_network, d_network, association)
# c_G, c_S, c_R = prepare_graph_data(c_adj)
# d_G, d_S, d_R = prepare_graph_data(d_adj)
# c_feature_dim = c_features.shape[1]
# d_feature_dim = d_features.shape[1]
# c_args.hidden_dims = [c_feature_dim] + c_args.hidden_dims
# d_args.hidden_dims = [d_feature_dim] + d_args.hidden_dims
# c_gate_trainer = GATETrainer(c_args)
# c_gate_trainer(c_G, c_features, c_S, c_R)
# c_embeddings, _ = c_gate_trainer.infer(c_G, c_features, c_S, c_R)
# tf.reset_default_graph()
# print('first finished')
# d_gate_trainer = GATETrainer(d_args)
# d_gate_trainer(d_G, d_features, d_S, d_R)
# d_embeddings, _ = d_gate_trainer.infer(d_G, d_features, d_S, d_R)
# print('second finished')