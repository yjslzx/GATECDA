import numpy as np


def GIP_kernel(Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(Asso_RNA_Dis):
    # calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r


def get_syn_sim(A, seq_sim, str_sim, mode):
    """

    :param A:
    :param seq_sim:
    :param str_sim:
    :param mode: 0 = GIP kernel sim
    :return:
    """

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((A.shape[0], A.shape[0]))
    syn_d = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2

    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2

    return syn_c, syn_d


def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy


# ######################################################################################################################


def get_syn_sim_circ_drug(A, seq_sim, str_sim, k1, k2):
    disease_sim1 = str_sim
    circRNA_sim1 = seq_sim

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)
    # miRNA_sim1 = GIP_m_sim
    m1 = new_normalization(circRNA_sim1)
    m2 = new_normalization(GIP_c_sim)

    Sm_1 = KNN_kernel(circRNA_sim1, k1)
    Sm_2 = KNN_kernel(GIP_c_sim, k1)
    Pm = circRNA_updating(Sm_1, Sm_2, m1, m2)
    Pm_final = (Pm + Pm.T) / 2

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(GIP_d_sim)

    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating(Sd_1, Sd_2, d1, d2)
    Pd_final = (Pd + Pd.T) / 2

    return Pm_final, Pd_final


def new_normalization(w):
    m = w.shape[0]
    p = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1 / 2
            elif np.sum(w[i, :]) - w[i, i] > 0:
                p[i][j] = w[i, j] / (2 * (np.sum(w[i, :]) - w[i, i]))
    return p


def KNN_kernel(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])
        for j in sort_index[n - k:n]:
            if np.sum(S[i, sort_index[n - k:n]]) > 0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
    return S_knn


def circRNA_updating(S1, S2, P1, P2):
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    return P


def disease_updating(S1, S2, P1, P2):
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    return P


# ====================================================================
def skf_normalization(w):
    row_sum = np.sum(w, axis=0)
    p = (w / row_sum).T
    return p


def skf(A, seq_sim, str_sim, k1, k2):
    disease_sim1 = str_sim
    circRNA_sim1 = seq_sim

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)
    # miRNA_sim1 = GIP_m_sim
    m1 = skf_normalization(circRNA_sim1)
    m2 = skf_normalization(GIP_c_sim)

    Sm_1 = KNN_kernel(circRNA_sim1, k1)
    Sm_2 = KNN_kernel(GIP_c_sim, k1)
    Pm = skf_updating(Sm_1, Sm_2, m1, m2, 0.1)
    nei_weight1 = neighborhood_Com(Pm, k1)
    Pm_final = Pm * nei_weight1

    d1 = skf_normalization(disease_sim1)
    d2 = skf_normalization(GIP_d_sim)

    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(GIP_d_sim, k2)
    Pd = skf_updating(Sd_1, Sd_2, d1, d2, 0.1)
    nei_weight2 = neighborhood_Com(Pd, k2)
    Pd_final = Pd * nei_weight2

    return Pm_final, Pd_final


def skf_updating(S1, S2, P1, P2, alpha):
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        P111 = alpha * np.dot(np.dot(S1, P2), S1.T) + (1 - alpha) * P2
        P111 = new_normalization(P111)
        P222 = alpha * np.dot(np.dot(S2, P1), S2.T) + (1 - alpha) * P1
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    return P


def neighborhood_Com(sim, k):
    weight = np.zeros(sim.shape)

    for i in range(sim.shape[0]):
        iu = sim[i, :]
        iu_list = np.abs(np.sort(-iu))
        iu_nearest_list_end = iu_list[k - 1]
        for j in range(sim.shape[1]):
            ju = sim[:, j]
            ju_list = np.abs(np.sort(-ju))
            ju_nearest_list_end = ju_list[k - 1]

            if sim[i, j] >= iu_nearest_list_end and sim[i, j] >= ju_nearest_list_end:
                weight[i, j] = 1
                weight[j, i] = 1
            elif sim[i, j] < iu_nearest_list_end and sim[i, j] < ju_nearest_list_end:
                weight[i, j] = 0
                weight[j, i] = 0
            else:
                weight[i, j] = 0.5
                weight[j, i] = 0.5

    return weight
