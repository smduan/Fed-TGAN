from fedtgan.model import  Generator, Discriminator
import torch
from torch import optim
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from torch.nn import functional as F
from fedtgan.data_transformer import DataTransformer
from conf import conf
from similarity_test import table_similarity
import pandas as pd

class SingleGAN(object):

    def __init__(self,conf,train_df, test_df,cuda=True):

        self.train_df = train_df

        self._batch_size = conf['batch_size']

        self.local_epoch = conf['local_epochs']
        self.local_discriminator_steps = conf['local_discriminator_steps']

        self.gen_lr = conf['gen_lr']
        self.gen_weight_decay = conf['gen_weight_decay']
        self.dis_lr = conf['dis_lr']
        self.dis_weight_decay = conf['dis_weight_decay']

        self.discrete_columns = conf['discrete_columns'][conf['data_name']]
        self.max_clusters = conf['max_clusters']
        self.embedding_dim = conf['latent_size']

        self.generator_dim = conf['generator_dim']
        self.discriminator_dim = conf['discriminator_dim']
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self.device = torch.device(device)

    def sample_data(self):
        """
        :return: data sampling
        """

        data_size = len(self.train_data)
        if data_size > self._batch_size:

            index = np.random.randint(data_size, size=self._batch_size)

            return self.train_data[index]
        else:
            return self.train_data
    def apply_activate(self,data):

        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
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

    def train(self):

        self.transformer = DataTransformer()
        self.transformer.fit(self.train_df,self.discrete_columns)

        self.train_data = self.transformer.transform(self.train_df)

        data_dim = self.transformer.output_dimensions
        print("data dimension: {}".format(data_dim))

        self.generator = Generator(self.embedding_dim, self.generator_dim, data_dim).to(self.device)

        self.discriminator = Discriminator(data_dim,self.discriminator_dim).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.gen_lr, betas=(0.5, 0.9),
            weight_decay=self.gen_weight_decay
        )

        optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.dis_lr,
            betas=(0.5, 0.9), weight_decay=self.dis_weight_decay
        )

        mean = torch.zeros(self._batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        # loss = torch.nn.BCELoss().to(self.device)

        training_step_per_epoch = max(len(self.train_data) // self._batch_size , 1)

        for i in range(self.local_epoch):

            # self.generator.train()
            # self.discriminator.train()
            for j in range(training_step_per_epoch):
                # taining discriminator
                for n_d in range(self.local_discriminator_steps):
                    noise = torch.normal(mean=mean, std=std)
                    real = self.sample_data()

                    fake = self.generator(noise)
                    fakeact = self.apply_activate(fake)

                    fake_critic = self.discriminator(fakeact)
                    # fake_label = torch.zeros(fake_critic.shape,device=self.device)
                    #
                    # fake_loss = loss(fake_critic, fake_label)


                    real = torch.from_numpy(real.astype('float32')).to(self.device)
                    # print("real shape: {}".format(real.shape))

                    real_critic = self.discriminator(real)

                    pen = self.discriminator.calc_gradient_penalty(real, fakeact, self.device)
                    # real_label = torch.ones(real_critic.shape, device=self.device)
                    # real_loss = loss(real_critic, real_label)

                    loss_d = -((torch.mean(real_critic)) - torch.mean(fake_critic))

                    # loss_d = fake_loss + real_loss

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # # training generator
                # if index % self.local_discriminator_steps == 0:
                noise = torch.normal(mean=mean, std=std)
                fake = self.generator(noise)
                fakeact = self.apply_activate(fake)
                fake_critic = self.discriminator(fakeact)
                # real_label = torch.ones(fake_critic.shape, device=self.device)
                # loss_g = loss(fake_critic, real_label)
                loss_g = -torch.mean(fake_critic)
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            print("Epoch {0}, loss G: {1}, loss D: {2}".format(i + 1, loss_g.detach().cpu(),
                                                               loss_d.detach().cpu()))

    @torch.no_grad()
    def sample(self, n):
        self.generator.eval()
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)
            if torch.cuda.is_available():
                fakez = fakez.cuda()
            fake = self.generator(fakez)
            fakeact = self.apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data)

if __name__ == "__main__":
    dis_columns = conf["discrete_columns"][conf["data_name"]]
    test_data = pd.read_csv(conf['test_dataset'][conf['data_name']])

    real_data = pd.read_csv(conf['train_dataset'][conf['data_name']])

    sg = SingleGAN(conf, real_data, test_data)
    sg.train()
    syn_data = sg.sample(1000)
    avg_jsd, avg_wd = table_similarity(syn_data, test_data, dis_columns)
    print("avg_jsd:{}".format(avg_jsd))
    print("avg_wd:{}".format(avg_wd))










