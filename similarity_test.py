import pandas as pd
from conf import conf
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import  MinMaxScaler
from ctgan import CTGAN


def table_similarity(syn_data, real_data, dis_columns):
    """

    :param syn_data:
    :param real_data:
    :param dis_columns:
    :return: compute the average Jensen-Shannon Divergence (JSD) and the average Wasserstein Distance (WD)
    """

    columns = real_data.columns

    jsd = []
    wd =[]
    for c in columns:

        if c in dis_columns:
            jsd.append(cal_jsd(syn_data[c], real_data[c]))
        else:
            wd.append(cal_wd(syn_data[c], real_data[c]))

    avg_jsd = sum(jsd) / len(jsd)
    avg_wd = sum(wd) / len(wd)

    return avg_jsd, avg_wd

def get_fre(data):
    cf = data.value_counts()
    cate = []
    fre = []
    for c, f in enumerate(cf):
        cate.append(c)
        fre.append(f)
    cate_fre = pd.DataFrame({'cate':cate, 'fre':fre})
    return cate_fre


def cal_jsd(syn,real):
    """
    :param syn:
    :param real:
    :return: compute the js distance for discrete columns between synthetic data and real data
    """
    syn_cf = get_fre(syn)
    real_cf = get_fre(real)

    if len(syn_cf) > len(real_cf):
        cate = syn_cf['cate'].tolist()
    else:
        cate = real_cf['cate'].tolist()

    syn_f = []
    real_f =[]
    for c in cate:
        s = syn_cf[syn_cf['cate']==c]['fre'].values
        if len(s) >0:
            syn_f.append(s[0])
        else:
            syn_f.append(0)

        f = real_cf[real_cf['cate']==c]['fre'].values
        if len(f)>0:
            real_f.append(f[0])
        else:
            real_f.append(0)

    return distance.jensenshannon(syn_f,real_f,base=2)


def cal_wd(syn,real):
    """
    :param syn:
    :param real:
    :return: the Wasserstein Distance for each continuous column
    """
    min_max_enc = MinMaxScaler(feature_range=(0, 1))
    syn = min_max_enc.fit_transform(syn.values.reshape(-1,1))
    real = min_max_enc.fit_transform(real.values.reshape(-1,1))

    return wasserstein_distance(syn.ravel(),real.ravel())

def ctgan_syn(real_data, dis_columns, num):


    ctgan = CTGAN(epochs=10,verbose=True)
    ctgan.fit(real_data, dis_columns)

    return ctgan.sample(num)


if __name__ == "__main__":

    dis_columns = conf["discrete_columns"][conf["data_name"]]
    test_data = pd.read_csv(conf['test_dataset'][conf['data_name']])

    real_data = pd.read_csv(conf['train_dataset'][conf['data_name']])

    avg_jsd, avg_wd = table_similarity(real_data,test_data,dis_columns)
    print("training data")
    print("avg_jsd:{}".format(avg_jsd))
    print("avg_wd:{}".format(avg_wd))

    syn_data = ctgan_syn(real_data, dis_columns, 1000)
    avg_jsd, avg_wd = table_similarity(syn_data, test_data, dis_columns)
    print("synthetic data")
    print("avg_jsd:{}".format(avg_jsd))
    print("avg_wd:{}".format(avg_wd))