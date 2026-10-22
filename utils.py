import os
from types import SimpleNamespace

import config
import numpy as np
import pandas as pd
from nilearn import connectome

cfg = SimpleNamespace(**vars(config))


def load_connectivity():
    dir = "/ABIDE1/rois_aal"
    label_path = "/ABIDE1/Phenotypic_V1_0b_preprocessed1 (871).csv"
    df = pd.read_csv(label_path)
    data_list = []
    data_dirs = df["FILE_ID"]

    for data_dir in data_dirs:
        data_dir = data_dir + "_rois_aal.1D"
        data = np.loadtxt(os.path.join(dir, data_dir))
        data = subject_connectivity(data)
        data_list.append(data)

    data_list = np.array(data_list, dtype=np.float32)

    return data_list


def load_data():
    labels, connectivities = get_dataset()
    connectivities = connectivities[:, :90, :90]
    connectivities = sort_matrix_by_order(connectivities)
    hypergraphs = get_hypergraphs(connectivities)
    return connectivities, hypergraphs, labels


def get_dataset():
    labels, all_data = [], []
    excel_dir = "ADHD/ADHD/ADHD_labels.csv"
    excel = pd.read_csv(excel_dir)
    IDs = excel["ID"].tolist()
    IDs = [str(x) for x in IDs]

    base_dir = r"ADHD/ADHD/ADHD/ADHD_all"
    badData = [
        "0010127",
        "0015057",
        "0015032",
        "0010027",
        "0015001",
        "0015026",
        "0010016",
        "0010055",
        "0010105",
        "0015004",
        "0015018",
        "0015027",
        "0015011",
        "0010098",
        "0015052",
        "0015036",
        "0015016",
    ]
    for i, id in enumerate(IDs):
        if len(id) == 5:
            id = "00" + id
        if id not in badData:
            file_path = os.path.join(
                base_dir, id, f"sfnwmrda{id}_session_1_rest_1_aal_TCs.1D"
            )
            df = pd.read_csv(file_path, sep="\t", header=None)
            timeseries = df.iloc[1:, 2:].values
            timeseries = np.array(timeseries)
            timeseries = timeseries.astype(float)
            this_FC = subject_connectivity(timeseries, "correlation")
            all_data.append(this_FC)
            label = excel["DX"][i]
            if label > 1:
                label = 1
            labels.append(label)
    all_data = np.array(all_data)
    labels = np.array(labels)
    print("all_data", all_data.shape)

    return labels, all_data


def subject_connectivity(timeseries, kind="correlation"):
    if kind in ["tangent", "partial correlation", "correlation"]:
        conn_measure = connectome.ConnectivityMeasure(
            kind=kind, standardize="zscore_sample"
        )
        connectivity = conn_measure.fit_transform([timeseries])[0]
    return connectivity


def calculate_dist_matrix(features):
    sum_square = np.sum(np.square(features), axis=1, keepdims=True)  # [V, 1]
    dist_matrix = sum_square + sum_square.T - 2 * np.dot(features, features.T)
    dist_matrix = np.sqrt(np.maximum(dist_matrix, 0))
    return dist_matrix


def calculate_hypergraph_matrix(features):
    dist_matrix = calculate_dist_matrix(features)

    V = dist_matrix.shape[0]
    hyperedges = []
    for i in range(V):
        nearest_indices = np.argsort(dist_matrix[i])[1 : cfg.k + 1]
        hyperedges.append([i] + nearest_indices.tolist())

    E = len(hyperedges)
    hyper_incidence_matrix = np.zeros((V, E))
    for e_idx, edge in enumerate(hyperedges):
        for node in edge:
            hyper_incidence_matrix[node, e_idx] = 1

    return hyper_incidence_matrix


def get_hypergraphs(features):
    hypergraphs = []
    for feature in features:
        hypergraph = calculate_hypergraph_matrix(feature)
        hypergraphs.append(hypergraph)
    hypergraphs = np.array(hypergraphs, dtype=np.float32)
    return hypergraphs


def load_datas():
    if cfg.dataset == "ADNC":
        connectivities = np.load("ADNC/fmri.npy")
        hypergraphs = get_hypergraphs(connectivities)
        labels = np.load("ADNC/labels.npy")
        return connectivities, hypergraphs, labels
    else:
        return load_data()


def read_order_from_txt(file_path):
    order = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            order.append(int(parts[-3]))
    return np.array(order)


def sort_matrix_by_order(matrix):
    order = read_order_from_txt("New_Node_AAL90.node")
    sorted_matrix = np.zeros_like(matrix)

    sorted_matrix[:, :16] = matrix[:, order == 0]
    sorted_matrix[:, 16:22] = matrix[:, order == 1]
    sorted_matrix[:, 22:36] = matrix[:, order == 2]
    sorted_matrix[:, 36:56] = matrix[:, order == 3]
    sorted_matrix[:, 56:64] = matrix[:, order == 4]
    sorted_matrix[:, 64:72] = matrix[:, order == 5]
    sorted_matrix[:, 72:76] = matrix[:, order == 6]
    sorted_matrix[:, 76:] = matrix[:, order == 7]

    sorted_matrix[:, :, :16] = matrix[:, :, order == 0]
    sorted_matrix[:, :, 16:22] = matrix[:, :, order == 1]
    sorted_matrix[:, :, 22:36] = matrix[:, :, order == 2]
    sorted_matrix[:, :, 36:56] = matrix[:, :, order == 3]
    sorted_matrix[:, :, 56:64] = matrix[:, :, order == 4]
    sorted_matrix[:, :, 64:72] = matrix[:, :, order == 5]
    sorted_matrix[:, :, 72:76] = matrix[:, :, order == 6]
    sorted_matrix[:, :, 76:] = matrix[:, :, order == 7]
    return sorted_matrix
