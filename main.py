from conf import conf
import torch

from fedtgan.server import Server, StatisticalAggregation
from fedtgan.client import Client
from fedtgan.data_transformer import DataTransformer

from fedtgan.model import Discriminator, Generator,weights_init_normal
from utils import get_data
import copy
from similarity_test import table_similarity

def synthesize(n_sample):

    train_datasets, test_dataset, columns= get_data()
    print("data partitiorn done !")

    clients = {}
    clients_num = {}

    for key in train_datasets.keys():
        clients[key] = Client(conf,train_datasets[key])
        clients_num[key] = len(train_datasets[key])

    print("clients initialization done !")

    # federated feature encoding
    clients_categorical = {}
    clients_gmm = {}
    for key in clients.keys():
        cate_frequency, con_gmm = clients[key].compute_local_statistics()
        clients_categorical[key] = copy.deepcopy(cate_frequency)
        clients_gmm[key] = copy.deepcopy(con_gmm)

    print("local statistics aggregating ...")
    sa = StatisticalAggregation(conf, clients_categorical, clients_gmm, clients_num)
    vir_data = sa.construct_vir_data()
    #order the column
    vir_data = vir_data[columns]

    #global data transformer
    transformer = DataTransformer()
    transformer.fit(vir_data, conf['discrete_columns'][conf['data_name']])

    #input dimension
    data_dim = transformer.output_dimensions

    print('data dimensions :{}'.format(data_dim))

    print("lcoal data encoding ....")
    for key in clients.keys():
        clients[key].data_encoding(transformer)

    ##aggregate weight
    #compute the table-wise similarity weights
    client_weight = {}
    if conf["is_init_avg"]:
        ##fedavg
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)
    else:
        jsd = sa.compute_jsd_matrix()
        wd = sa.compute_wd_matrix(vir_data)
        new_weight = sa.compute_new_weight(jsd,wd)
        for key in train_datasets.keys():
            client_weight[key] = new_weight[key]

        print("new weight = {}".format(new_weight))
    print('weight init done !')


    clients_dis = {}
    clients_gen = {}

    #init models
    generator = Generator(conf['latent_size'],conf['generator_dim'],data_dim)
    discriminator = Discriminator(data_dim,conf['discriminator_dim'])
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    #init server
    server = Server(discriminator,generator)

    #init client model
    for key in train_datasets.keys():
        clients[key].init_model(copy.deepcopy(server.global_discriminator), copy.deepcopy(server.global_generator))

    #federated training
    for e in range(conf['global_epochs']):

        for key in clients.keys():
            print("client {0} training in epoch {1}".format(key,e))
            discriminator_k, generator_k = clients[key].local_train(server.global_discriminator, server.global_generator)
            clients_dis[key] = copy.deepcopy(discriminator_k)
            clients_gen[key] = copy.deepcopy(generator_k)

        #weight aggregate
        server.model_aggregate(clients_dis,clients_gen,client_weight)

    # data similarity tests
    syn_data = server.sample(n_sample,transformer,conf)
    avg_jsd, avg_wd = table_similarity(syn_data,test_dataset,conf["discrete_columns"][conf["data_name"]])
    print("epoch {0}, avg_jsd {1}, avg_wd {2}".format(e,avg_jsd,avg_wd))
    return syn_data


if __name__ == "__main__":

    syn_data = synthesize(1000)

    syn_data.to_csv(conf['syn_data'][conf['data_name']], index=False)