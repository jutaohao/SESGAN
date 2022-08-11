"""Sesgan
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
from collections import OrderedDict
import time

import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from lib.models.networks import NetG, weights_init
from lib.loss import l2_loss
from lib.evaluate import evaluate
from lib.models.basemodel import BaseModel

##
class Sesgan(BaseModel):
    """Sesgan Class
    """

    @property
    def name(self): return 'Sesgan'

    def __init__(self, opt, data):
        super(Sesgan, self).__init__(opt, data)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0


        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)

        self.netg.apply(weights_init)
        print(self.netg)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        self.fake, self.latent_i,self.pred_real, self.feat_real,self.pred_fake, self.feat_fake= self.netg(self.input)


    def backward_g(self):

        self.err_d_fake = self.l_bce(self.pred_fake,self.fake_label)
        self.err_d_real = self.l_bce(self.pred_real,self.real_label)
        self.err_g_lat = self.l_adv(self.feat_fake, self.latent_i)
        self.err_g = self.err_g_lat * self.opt.w_feat - (self.err_d_fake + self.err_d_real) * self.opt.w_adv
        self.err_g.backward(retain_graph=True)

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    ##
    def test(self):
        """ Test Sesgan model.

        Args:
            data ([type]): data for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']
                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            for ts in range(0, 1):
                self.opt.phase = 'test'
                # Create big error tensor for the test set.
                self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
                self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,    device=self.device)
                self.latent_i  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
                self.latent_o  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
                # print("   Testing model %s." % self.name)
                self.times = []
                self.total_steps = 0
                epoch_iter = 0
                for i, data in enumerate(self.data.valid, 0):
                    self.total_steps += self.opt.batchsize
                    epoch_iter += self.opt.batchsize
                    time_i = time.time()
                    self.set_input(data)
                    fake, latent_i, pred_real, feat_real, pred_fake, feat_fake = self.netg(self.input)
                    si = self.input.size()

                    if ts == 0:
                        error1 = torch.mean(torch.pow((latent_i - feat_fake), 2), dim=1)
                        error = error1
                    elif ts == 1:
                        rec1 = (self.input - fake).view(si[0], si[1] * si[2] * si[3])
                        rec1 = torch.mean(torch.pow(rec1, 2), dim=1).view(si[0], 1, 1)
                        error = rec1
                    elif ts == 2:
                        error1 = torch.mean(torch.pow((latent_i - feat_real), 2), dim=1)
                        error = error1
                    elif ts == 3:
                        error1 = torch.mean(torch.pow((feat_fake - feat_real), 2), dim=1)
                        error = error1
                    time_o = time.time()
                    self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                    self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                    self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                    self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = feat_fake.reshape(error.size(0), self.opt.nz)
                    self.times.append(time_o - time_i)

                # Measure inference time.
                self.times = np.array(self.times)
                self.times = np.mean(self.times[:100] * 1000)

                # Scale error vector between [0, 1]
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
                auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
                performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

