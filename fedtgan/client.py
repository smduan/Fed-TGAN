import torch
from torch import optim
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from torch.nn import functional as F

class Client(object):

    def __init__(self, conf, train_df, cuda=True):
        """
        client side
        """

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
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self.device = torch.device(device)

    def init_model(self,discriminator, generator):
        """
        :param discriminator:
        :param generator:
        :return:
        """

        self.local_discriminator = discriminator
        self.local_generator = generator

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


    def local_train(self, discriminator, generator):
        """
        :param discriminator:
        :param generator:
        :return:
        """

        #update the local generator and discriminator
        for name, param in discriminator.state_dict().items():
            self.local_discriminator.state_dict()[name].copy_(param.clone())

        for name, param in generator.state_dict().items():
            self.local_generator.state_dict()[name].copy_(param.clone())

        optimizerG = optim.Adam(
            self.local_generator.parameters(), lr=self.gen_lr, betas=(0.5, 0.9),
            weight_decay=self.gen_weight_decay
        )

        optimizerD = optim.Adam(
            self.local_discriminator.parameters(), lr=self.dis_lr,
            betas=(0.5, 0.9), weight_decay=self.dis_weight_decay
        )

        mean = torch.zeros(self._batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        training_step_per_epoch = max(len(self.train_data) // self._batch_size , 1)

        for i in range(self.local_epoch):

            # self.local_generator.train()
            # self.local_discriminator.train()
            for j in range(training_step_per_epoch):
                # taining discriminator
                for n_d in range(self.local_discriminator_steps):

                    noise = torch.normal(mean=mean, std=std)
                    fake = self.local_generator(noise)
                    fakeact = self.apply_activate(fake)

                    fake_critic = self.local_discriminator(fakeact)
                    # fake_label = torch.zeros(fake_critic.shape,device=self.device)
                    #
                    # fake_loss = loss(fake_critic, fake_label)

                    real = self.sample_data()
                    real = torch.from_numpy(real.astype('float32')).to(self.device)
                    # print("real shape: {}".format(real.shape))

                    real_critic = self.local_discriminator(real)

                    pen = self.local_discriminator.calc_gradient_penalty(real, fakeact, self.device)
                    # real_label = torch.ones(real_critic.shape, device=self.device)
                    # real_loss = loss(real_critic, real_label)

                    loss_d = -(torch.mean(real_critic) - torch.mean(fake_critic))

                    # loss_d = fake_loss + real_loss

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # # training generator
                # if index % self.local_discriminator_steps == 0:
                noise = torch.normal(mean=mean, std=std)
                fake = self.local_generator(noise)
                fakeact = self.apply_activate(fake)
                fake_critic = self.local_discriminator(fakeact)
                # real_label = torch.ones(fake_critic.shape, device=self.device)
                # loss_g = loss(fake_critic, real_label)
                loss_g = -torch.mean(fake_critic)
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            print("Epoch {0}, loss G: {1}, loss D: {2}".format(i + 1, loss_g.detach().cpu(),
                                                                       loss_d.detach().cpu()))

        return self.local_discriminator.state_dict(), self.local_generator.state_dict()

    def compute_local_statistics(self):
        """
        :return: compute the frequency of categorical columns and the gmm for continuous columns
        """
        columns = self.train_df.columns

        categorical = {}
        continuous = {}

        for c in columns:

            if c in self.discrete_columns:
                categorical[c] = self.categorical_frequency(c)
            else:
                continuous[c] = self.continuous_gmm(c)

        return categorical, continuous


    def categorical_frequency(self,column):
        """
        :param column: categorical column name
        :return: frequency of categories
        """

        fre = self.train_df[column].value_counts()
        categorical_frequency = {}
        for cat, value in zip(fre._index, fre.values):
            categorical_frequency[cat] = value
        return categorical_frequency

    def continuous_gmm(self,column):
        """
        :param column: continuous column
        :return: GMM of the continuous column
        """

        data = self.train_df[column].values

        gmm = BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gmm.fit(data.reshape(-1,1))

        return gmm

    def data_encoding(self,transformer):
        """
        :param transformer:
        :return: encode the local data by the global transformer
        """

        self.transformer = transformer

        self.train_data = self.transformer.transform(self.train_df)


    def sample_data(self):
        """
        :return: sample data for training
        """

        data_size = len(self.train_data)

        index = np.random.randint(data_size, size=self._batch_size)

        return self.train_data[index]
