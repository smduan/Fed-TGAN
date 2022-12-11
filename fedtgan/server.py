import torch
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import  MinMaxScaler
from torch.nn import functional as F

class Server(object):

    def __init__(self,discriminator, generator):
        """
        :param discriminator:
        :param generator:

        initialized generator and discriminator
        """

        self.global_discriminator = discriminator
        self.global_generator = generator


    def model_update(self,base_model, clients_model, weights):
        """
        :param base_model:
        :param clients_model:
        :param weights:
        :return: weight aggregating
        """

        new_model = {}

        #initial a new model
        for name, params in base_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        #update the new model with weights
        for key in clients_model.keys():
            for name, param in clients_model[key].items():
                new_model[name]= new_model[name] + clients_model[key][name] * weights[key]

        return new_model

    def model_aggregate(self, clients_dis, clients_gen, weights):
        """
        :param clients_dis:
        :param clients_gen:
        :param weights:
        :return: update the global GAN
        """

        global_dis = self.model_update(self.global_discriminator,clients_dis,weights)
        global_gen = self.model_update(self.global_generator,clients_gen, weights)

        #update the global discriminator and generator
        self.global_discriminator.load_state_dict(global_dis)
        self.global_generator.load_state_dict(global_gen)

    def apply_activate(self, transformer, data):

        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = F.gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    @torch.no_grad()
    def sample(self,n, transformer, conf):
        """
        :param n:
        :param transformer:
        :param conf:
        :return: generate synthetic data
        """
        self.global_generator.eval()
        steps = n // conf['batch_size'] + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(conf['batch_size'], conf['latent_size'])
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)
            if torch.cuda.is_available():
                fakez = fakez.cuda()
            fake = self.global_generator(fakez)
            fakeact = self.apply_activate(transformer, fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return transformer.inverse_transform(data)


##aggregate the local statistics from each client
class StatisticalAggregation(object):

    def __init__(self,conf, clients_frequency, clients_gmm, clients_num):
        """
        :param conf: parameters
        :param clients_frequency: frequency of each discrete column from each client
        :param clients_gmm: gmm model of each continuous column from each client
        :param clients_num: the number of samples of each client
        """

        # config parameters
        self.conf = conf

        self.clients_frequency = clients_frequency

        self.clients_gmm = clients_gmm
        self.client_num = clients_num

        # categorical classes of each discrete column
        self.global_categorical = {}
        # frequency of each categorical class
        self.global_frequency = {}

        self.frequency_aggregation()

    def frequency_aggregation(self):
        """
        :return: aggregate the frequency of categorical columns from each client
        """

        cate_columns = self.conf['discrete_columns'][self.conf['data_name']]
        if len(cate_columns) > 0:
            for c in cate_columns:
                cate = []
                fre = []
                for key in self.clients_frequency.keys():
                    cate_k = list(self.clients_frequency[key][c].keys())
                    fre_k = list(self.clients_frequency[key][c].values())
                    cate.extend(cate_k)
                    fre.extend(fre_k)
                cate_c = list(set(cate))
                cate_fre = pd.DataFrame({'cate':cate,'fre':fre})
                total_fre = []
                for ca in cate_c:
                    fre_ca = cate_fre[cate_fre['cate'] == ca]['fre'].sum()
                    total_fre.append(fre_ca)

                self.global_categorical[c] = cate_c
                self.global_frequency[c] = total_fre


    def construct_vir_cate_data(self):
        """
        :return: construct a virtual discrete column, which has the same frequency with the global data
        """

        vir_cate_data = {}
        if self.global_categorical and self.global_frequency:
            for c in self.global_categorical:
                cate = np.array(self.global_categorical[c])
                fre = self.global_frequency[c]

                data_c = cate.repeat(fre)
                vir_cate_data[c] = data_c
        else:
            vir_cate_data = None

        return pd.DataFrame(vir_cate_data)

    def construct_vir_con_data(self):
        """
        :return: construct virtual continuous data by gmm.

        """

        clients = list(self.clients_gmm.keys())
        con_columns = list(self.clients_gmm[clients[0]].keys())
        if len(con_columns) > 0:
            vir_con_data = {}
            for c in con_columns:
                data_c = []
                for key in self.clients_gmm.keys():
                    gmm = self.clients_gmm[key][c]
                    samples = gmm.sample(n_samples=self.client_num[key])
                    data_c.extend(list(samples[0].reshape(-1)))
                vir_con_data[c] = data_c
            return pd.DataFrame(vir_con_data)
        else:
            return None

    def construct_vir_data(self):
        """
        :return: construct a virtual data that has the same frequency and gmm with global real data.

        The constructed virtual data is used to train a global transformer.

        """

        vir_cate_data = self.construct_vir_cate_data()
        vir_con_data = self.construct_vir_con_data()

        return pd.concat([vir_cate_data,vir_con_data], axis=1)

    def compute_column_frequency(self):
        """
        :return: each column frequency with the same sequence
        {
            "column1":[[1,0,3,4,5],[1,2,3,4,5]],
            "column2":[[1,2,3,4,5],[1,2,3,4,5]]
        }
        """

        cate_columns = self.conf['discrete_columns'][self.conf['data_name']]

        if len(cate_columns)>0:
            client_class = {}
            for c in cate_columns:
                cate_c = self.global_categorical[c]
                client_class_fre = []

                for key in self.clients_frequency.keys():
                    client_fre= []
                    cate_k = list(self.clients_frequency[key][c].keys())
                    for cl in cate_c:
                        if cl in cate_k:
                            client_fre.append(self.clients_frequency[key][c][cl])
                        else:
                            client_fre.append(0)

                    client_class_fre.append(client_fre)

                client_class[c]=client_class_fre
        else:
            client_class = None

        return client_class

    def compute_jsd_matrix(self):
        """

        :return: JSD matrix of discrete columns (K,L),K is the number of clients, L is the number of discrete columns
        """

        client_class = self.compute_column_frequency()

        if client_class:
            number_clients = len(self.clients_frequency)
            dis_columns = list(client_class.keys())
            number_dis_columns = len(dis_columns)
            jsd = np.zeros((number_clients,number_dis_columns),dtype=float)

            for i in range(number_dis_columns):
                c= dis_columns[i]
                frequency_c =client_class[c]
                for j in range(number_clients):
                    frequency_j = frequency_c[j]
                    jsd[j][i] = distance.jensenshannon(frequency_j,self.global_frequency[c])

        else:
            jsd = None

        return jsd

    def compute_wd_matrix(self, vir_data):

        clients = list(self.clients_gmm.keys())
        con_columns = list(self.clients_gmm[clients[0]].keys())
        if len(con_columns) > 0:

            vir_con_data = vir_data[con_columns]
            number_clients = len(self.clients_gmm)
            number_con_columns = len(con_columns)
            wd = np.zeros((number_clients,number_con_columns), dtype=float)

            min_max_enc = MinMaxScaler(feature_range=(0, 1))
            for i in range(number_con_columns):
                col = con_columns[i]

                for j in range(number_clients):
                    client_name = clients[j]
                    gmm = self.clients_gmm[client_name][col]
                    samples = gmm.sample(n_samples=self.client_num[client_name])

                    x = min_max_enc.fit_transform(samples[0].reshape(-1,1))
                    y = min_max_enc.fit_transform(vir_con_data[col].values.reshape(-1,1))

                    wd[j][i] = wasserstein_distance(x.ravel(), y.ravel())
        else:

            wd = None
        return wd

    def compute_new_weight(self, jsd, wd):
        """
        :return: table wise similarity weight []
        """
        if jsd is not None and wd is not None:
            jsd_wd = np.concatenate((jsd,wd),axis=1)
        elif jsd is not None and wd is None:
            jsd_wd = jsd
        else:
            jsd_wd = wd


        # step 1
        sum_s1 = np.sum(jsd_wd, axis=0)

        #if a value in sum_s1 is 0, which indicates that the distribution in each client is equal to the global data
        # result in nan
        index = np.where(sum_s1 == 0)
        sum_s1 = np.delete(sum_s1, index)
        jsd_wd = np.delete(jsd_wd, index, axis=1)

        ss = jsd_wd / sum_s1

        #step 2
        ss = np.sum(ss, axis=1)

        #step 3
        sum_ss = np.sum(ss)

        #if a value in sum_ss is 0
        #will result in nan
        index = np.where(sum_ss==0)
        sum_ss = np.delete(sum_ss,index)
        ss = np.delete(ss, index)

        clients_number = np.array(list(self.client_num.values()))
        total_number = np.sum(clients_number)
        sd = (clients_number / total_number)*(1-ss/sum_ss)

        #step 4
        new_weight = self.softmax(sd)

        return new_weight


    def softmax(self,x):
        """
        :param x:
        :return: softmax
        """

        row_max = np.max(x)
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        return x_exp/x_sum
