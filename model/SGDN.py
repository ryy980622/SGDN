# -*- coding: utf-8 -*-

import argparse
import math
import random
import string
from abc import ABC

import torch
import torch as th
from torch.nn import init

from data_disentangle import MovieLens
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from util import *
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter
import faiss

def config():
    parser = argparse.ArgumentParser(description='RGC')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--model_save_path', type=str, help='The model saving path')
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--review_feat_size', type=int, default=64)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--gcn_dropout', type=float, default=0.8)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--train_classification', type=bool, default=False)
    parser.add_argument('--num_factor', type=int, default=2)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--num_pos', type=int, default=10)
    parser.add_argument('--lamda', type=float, default=0.005)


    args = parser.parse_args()
    args.model_short_name = 'SGDN'

    args.dataset_name = 'Office_Products_5'
    args.dataset_path = '/home/ryy/code/GNN/ReviewGraph/data/' + args.dataset_name + '/' + args.dataset_name + '.json'
    args.train_max_iter = 2000


    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')

    # configure save_fir to save all the info
    if args.model_save_path is None:
        args.model_save_path = 'log/' \
                               + args.model_short_name \
                               + '_' + args.dataset_name \
                               + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=2)) \
                               + '.pkl'
    if not os.path.isdir('log'):
        os.makedirs('log')

    args.gcn_agg_units = args.review_feat_size*1
    args.gcn_out_units = args.review_feat_size*1

    return args


class GCMCGraphConv(nn.Module, ABC):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 num_factor,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.k = k
        self.device = device

        self.dropout = nn.Dropout(dropout_rate)
        self.review_w = nn.Linear(self._out_feats*num_factor, 64//num_factor, bias=False)
        self.node_w = nn.Linear(64//num_factor, 64//num_factor, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.review_w.weight)
        init.xavier_uniform_(self.node_w.weight)

    def forward(self, graph, feat, weight=None):

        with graph.local_scope():
            e = graph.canonical_etypes[0][1]
            if len(e)>1:
                s = 'h_' + e[-1]
            else:
                s = 'h_' + e[-1]
            graph.srcdata['h'] = self.node_w(feat[0][s])
            review_feat = graph.edata['review_feat']
            graph.edata['rf'] = self.review_w(review_feat)
            graph.update_all(lambda edges: {'m': (edges.src['h']
                                                  + edges.data['rf']) * self.dropout(edges.data['w'])},
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

        return rst


class GCMCLayer(nn.Module, ABC):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 num_rating,
                 msg_units,
                 out_units,
                 k,
                 num_factor,
                 aggregate='sum',
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.k = k  # 第k个factor
        sub_conv = {}
        self.aggregate = aggregate  # stack or sum
        self.num_user = user_in_units
        self.num_movie = movie_in_units
        self.num_factor = num_factor
        self.num_rating = num_rating
        self.eta = nn.Parameter(torch.FloatTensor(self.rating_vals[-1], self.num_rating))

        for rating in rating_vals:

            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            self.W_r = None
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             msg_units,
                                             k,
                                             num_factor,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                 msg_units,
                                                 k,
                                                 num_factor,
                                                 device=device,
                                                 dropout_rate=dropout_rate)

        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate=self.aggregate)
        self.agg_act = nn.LeakyReLU(0.1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, prototypes_all, review_feat_dic, ufeat, ifeat, feat_dic):
        tau =0.5
        exp_anchor_dot = {}

        num_user, num_movie = self.num_user, self.num_movie
        norm_user_sum, norm_movie_sum = torch.zeros(num_user).to(self.device), torch.zeros(num_movie).to(self.device)
        for u, e, v in graph.canonical_etypes:
            e_s, e_sum = 'h_' + e[-1], 'h_sum' + e[-1]
            rating = int(e[-1])-1
            row, col = graph[(u, e, v)].edges()[0].to(torch.int64), graph[(u, e, v)].edges()[1].to(torch.int64)
            row_feat, col_feat = F.normalize(feat_dic[self.k][u][e_s][row], dim=1), F.normalize(feat_dic[self.k][v][e_s][col], dim=1)
            row_all = feat_dic[u][e_sum][row]
            col_all = feat_dic[v][e_sum][col]
            sim_k = (row_feat*col_feat).sum(1) / tau
            sim_all = (row_all*col_all).sum(2) / tau
            exp_sim = torch.exp(sim_k) / torch.exp(sim_all).sum(1)



            review_feat_all = review_feat_dic[e[-1]]
            review_feat = review_feat_dic[e[-1]][:, self.k, :]
            graph.edges[e].data['review_feat'] = review_feat
            prototypes = prototypes_all
            anchor_dot_k = torch.matmul(review_feat, prototypes[self.k]) / tau
            anchor_dot_all = (review_feat_all * prototypes).sum(2) / tau

            exp_anchor_dot_k = torch.exp(anchor_dot_k) / torch.exp(anchor_dot_all).sum(1)
            exp_anchor_dot_k = F.sigmoid(self.eta[rating][:exp_anchor_dot_k.shape[0]])*exp_anchor_dot_k + (1-F.sigmoid(self.eta[rating][:exp_anchor_dot_k.shape[0]]))*exp_sim
            exp_anchor_dot[u+e+v] = exp_anchor_dot_k

            if u == 'movie':
                norm_movie = scatter(exp_anchor_dot_k, row, dim=0, dim_size=num_movie, reduce='sum')
                norm_user = scatter(exp_anchor_dot_k, col, dim=0, dim_size=num_user, reduce='sum')
            else:
                norm_user = scatter(exp_anchor_dot_k, row, dim=0, dim_size=num_user, reduce='sum')
                norm_movie = scatter(exp_anchor_dot_k, col, dim=0, dim_size=num_movie, reduce='sum')
            norm_user_sum += norm_user
            norm_movie_sum += norm_movie
        norm_user_sum, norm_movie_sum = norm_user_sum/2, norm_movie_sum/2
        int_dist = []
        for u, e, v in graph.canonical_etypes:
            exp_anchor_dot_k = exp_anchor_dot[u+e+v]

            row, col = graph[(u, e, v)].edges()[0].to(torch.int64), graph[(u, e, v)].edges()[1].to(torch.int64)
            if u=='movie':
                n_ij = torch.sqrt(norm_movie_sum[row] * norm_user_sum[col])
            else:
                n_ij = torch.sqrt(norm_user_sum[row] * norm_movie_sum[col])

            graph.edges[e].data['w'] = (exp_anchor_dot_k/ n_ij).unsqueeze(1)
            if u == 'movie':
                int_dist.append(graph.edges[e].data['w'])
        int_dist = torch.cat(int_dist, dim=0)
        out_feats = self.conv(graph, feat_dic[self.k])
        ufeat = out_feats['user']
        ifeat = out_feats['movie']

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat, int_dist


def cal_c_loss(h_fea1, h_fea2, int_dist, rating_split, k):
    tau = 0.2
    pos = 0
    c_loss = 0
    for num in rating_split:
        h_fea1_rating = F.normalize(h_fea1[pos: pos + num], dim=1)
        h_fea2_rating = F.normalize(h_fea2[pos: pos + num], dim=1)
        int_dist_rating = int_dist[pos: pos + num]
        sim_matrix = torch.matmul(int_dist_rating, int_dist_rating.transpose(0,1))
        _, indices = torch.topk(sim_matrix, dim=1, k=k)
        pos_fea = h_fea2_rating[indices]
        pos_score = (pos_fea.transpose(0,1)*h_fea1_rating).sum(dim=2).transpose(0, 1)
        pos_score = torch.exp(pos_score / tau).sum(dim=1)

        rand_index = torch.randperm(num, out=None, dtype=torch.int64)[:2048]
        ttl_score = torch.matmul(h_fea1_rating, h_fea2_rating[rand_index].transpose(0, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / tau), axis=1)
        c_loss += - torch.mean(torch.log(pos_score / ttl_score))
        pos += num

    return c_loss
class MLPPredictor(nn.Module, ABC):
    def __init__(self,
                 in_units,
                 rating_split,
                 num_classes,
                 num_factor,
                 dropout_rate=0.0):
        super(MLPPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 64, bias=False),
        )
        self.predictor = nn.Linear(64, num_classes, bias=False)
        self.rating_split = rating_split
        self.num_factor = num_factor
        self.reset_parameters()
        self.w = nn.Linear(in_units * 2//num_factor, 1)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        if self.num_factor==1:
            h_fea = self.linear(th.cat([h_u, h_v], dim=1))
            score = self.predictor(h_fea)
            return {'score': score, 'feat': h_fea}
        h_u, h_v = h_u.view(h_u.shape[0], self.num_factor, -1), h_v.view(h_v.shape[0], self.num_factor, -1)
        x = []
        for k in range(self.num_factor):
            h_fea_k = torch.cat([h_u[:, k, :], h_v[:, k, :]], dim=1)
            x.append(h_fea_k.unsqueeze(1))
        x = torch.cat(x, dim=1)
        x= x.reshape(x.shape[0], -1)
        h_fea = self.linear(x)
        score = self.predictor(h_fea).squeeze()
        return {'score': score, 'feat': h_fea}

    def forward(self, graph, ufeat, ifeat):
        graph.nodes['movie'].data['h'] = ifeat
        graph.nodes['user'].data['h'] = ufeat

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            return graph.edata['score'], graph.edata['feat']


class Net(nn.Module, ABC):
    def __init__(self, params, num_user, num_item, review_feat_dic, num_rating, rating_split):
        super(Net, self).__init__()
        self._act = get_activation(params.model_activation)
        self.encoders = [[] for _ in range(params.num_layer)]
        self.num_factor = params.num_factor
        self.num_user = num_user
        self.num_item = num_item
        self.num_rating = num_rating
        self.rating_vals = params.rating_vals
        self.rating_split = rating_split
        self.dim = params.gcn_out_units
        self.device = params.device
        self.num_layer = params.num_layer
        self.dropout = nn.Dropout(params.gcn_dropout)
        for l in range(params.num_layer):
            for k in range(params.num_factor):
                if l < params.num_layer-1:
                    aggr = 'stack'
                else:
                    aggr = 'sum'
                self.encoder = GCMCLayer(params.rating_vals,
                                         params.src_in_units,
                                         params.dst_in_units,
                                         self.num_rating,
                                         params.gcn_agg_units//self.num_factor,
                                         params.gcn_out_units//self.num_factor,
                                         k,
                                         self.num_factor,
                                         aggr,
                                         dropout_rate=params.gcn_dropout,
                                         device=params.device).to(params.device)
                self.encoders[l].append(self.encoder)
        for l in range(params.num_layer):
            for i, encoder in enumerate(self.encoders[l]):
                self.add_module('encoder_{}'.format(l*self.num_factor+i), encoder)
        self.ufeats = nn.ParameterDict()
        self.ifeats = nn.ParameterDict()
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                self.ufeats[str(r*len(self.rating_vals)+k)] = nn.Parameter(th.Tensor(num_user, params.gcn_out_units//self.num_factor))
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                self.ifeats[str(r*len(self.rating_vals)+k)] = nn.Parameter(th.Tensor(num_item, params.gcn_out_units//self.num_factor))
        self.prototypes = nn.Parameter(th.Tensor(self.num_factor, params.review_feat_size)).to(params.device)
        self.init_prot(review_feat_dic)
        self.rfcs = [nn.Linear(params.review_feat_size, params.review_feat_size).to(params.device) for _ in range(self.num_factor)]
        for i, fc in enumerate(self.rfcs):
            self.add_module('rfc_{}'.format(i), fc)


        self.decoder = MLPPredictor(in_units=params.gcn_out_units, rating_split=self.rating_split,
                                        num_classes=1, num_factor=self.num_factor, dropout_rate=0.0).to(params.device)
        self.reset_parameters()
    def init_prot(self, review_feat_dic):
        """Run K-means algorithm to get k clusters of the input tensor x
                """
        prototypes, reviews = [], []
        for rating in self.rating_vals:
            rating = to_etype_name(rating)
            review_feat = review_feat_dic[rating]
            # review_feat = review_feat.detach().cpu().numpy()
            reviews.append(review_feat)
        review_feat = torch.cat(reviews, dim=0).detach().cpu().numpy()
        kmeans = faiss.Kmeans(d=self.dim, k=self.num_factor, gpu=False)
        kmeans.train(review_feat)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(review_feat, 1)
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)
        self.prototypes.data = centroids
    def reset_parameters(self):
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                init.xavier_uniform_(self.ufeats[str(r*len(self.rating_vals)+k)])
        for r in range(len(self.rating_vals)):
            for k in range(self.num_factor):
                init.xavier_uniform_(self.ifeats[str(r*len(self.rating_vals)+k)])
        for i, rfeat in enumerate(self.rfcs):
            init.xavier_uniform_(self.rfcs[i].weight)
        init.xavier_uniform_(self.prototypes)
    def prepare_graph(self, l, graphs, user_out, movie_out):
        feat_dic_all = {'user': {}, 'movie': {} }
        user_sum, item_sum = [[] for _ in range(len(self.rating_vals))], [[] for _ in range(len(self.rating_vals))]
        for k in range(self.num_factor):

            graph = graphs[k]

            dic = {'user': {}, 'movie': {}}

            for u, e, v in graph.canonical_etypes:
                rating = int(e[-1]) - 1
                if u == 'user':
                    s = 'h_' + str(rating+1)
                    if l == 0:
                        dic['user'][s] = self.ufeats[str(rating*len(self.rating_vals)+k)]
                    else:
                        dic['user'][s] = user_out[k][:, rating, :]
                    user_sum[rating].append(dic['user'][s].unsqueeze(1))
                else:
                    s = 'h_' + str(rating+1)
                    if l == 0:
                        dic['movie'][s] = self.ifeats[str(rating*len(self.rating_vals)+k)]
                    else:
                        dic['movie'][s] = movie_out[k][:, rating, :]
                    item_sum[rating].append(dic['movie'][s].unsqueeze(1))
            feat_dic_all[k] = dic
        for rating in range(len(self.rating_vals)):
            feat_dic_all['user']['h_sum'+str(rating+1)] = F.normalize(torch.cat(user_sum[rating], dim=1), dim=2)
            feat_dic_all['movie']['h_sum'+str(rating+1)] = F.normalize(torch.cat(item_sum[rating], dim=1), dim=2)

        return feat_dic_all

    def forward(self, enc_graphs, dec_graph, review_feat_dic, save_graph=False):
        review_dic_fact = {}
        for rating in self.rating_vals:
            rating = to_etype_name(rating)
            review_feat = review_feat_dic[rating]
            temp = []
            for k in range(self.num_factor):
                review_feat_k = self._modules['rfc_'+str(k)](review_feat)
                dec_graph.edata['review_feat_'+str(k)] = self._modules['rfc_'+str(k)](dec_graph.edata['review_feat'])   # 加上dec_graph的review_feat
                temp.append(review_feat_k.unsqueeze(1))
            review_dic_fact[rating] = torch.cat(temp, dim=1)

        user_emb, item_emb, int_dists, feat_dic_all = [], [], [], {}
        user_out, movie_out, user_out_all, movie_out_all = [None for _ in range(self.num_factor)], [None for _ in range(
            self.num_factor)], [torch.zeros(self.num_user, self.dim//self.num_factor).to(self.device) for _
                                                            in range(self.num_factor)],[torch.zeros(self.num_item, self.dim//self.num_factor).to(self.device) for _
                                                            in range(self.num_factor)]
        for l in range(self.num_layer):
            feat_dic_all = self.prepare_graph(l, enc_graphs, user_out, movie_out)
            for k in range(self.num_factor):
                user_out[k], movie_out[k], int_dist = self._modules['encoder_'+str(l*self.num_factor+k)](enc_graphs[k], self.prototypes, review_dic_fact, self.ufeats, self.ifeats, feat_dic_all)
                if l !=self.num_layer-1:
                    user_out_all[k] += torch.sum(user_out[k], dim=1) * (1.0/(self.num_layer))
                    movie_out_all[k] += torch.sum(movie_out[k], dim=1) * (1.0 / (self.num_layer))
                else:
                    user_out_all[k] += user_out[k] * (1.0 / (self.num_layer))
                    movie_out_all[k] += movie_out[k] * (1.0 / (self.num_layer))
                    int_dists.append(int_dist)
        int_dists = torch.cat(int_dists, dim=1)

        for k in range(self.num_factor):
            user_emb.append(user_out_all[k])
            item_emb.append(movie_out_all[k])
            if k == 0:
                user_out, movie_out = user_out_all[k], movie_out_all[k]
            else:
                user_out = torch.cat([user_out, user_out_all[k]], dim=1)
                movie_out = torch.cat([movie_out, movie_out_all[k]], dim=1)


        pred_ratings, h_fea = self.decoder(dec_graph, user_out, movie_out)
        pred_ratings = pred_ratings.squeeze()


        if save_graph:
            for k in range(self.num_factor):
                dgl.save_graphs('data/'+config_args.dataset_name+'/graph_'+str(k)+'.dgl', enc_graphs[k])

        return pred_ratings, h_fea, int_dists, user_emb, item_emb, user_out, movie_out


def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graphs
        dec_graph = dataset.valid_dec_graph
        save_graph = False
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graphs
        dec_graph = dataset.test_dec_graph
        save_graph = True
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings, _,_, _, _, _, _ = net(enc_graph, dec_graph,
                           dataset.review_data_dict, save_graph=False)
        if args.train_classification:
            real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                                 nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
        else:
            rmse = ((pred_ratings - rating_values) ** 2.).mean().item()
        rmse = np.sqrt(rmse)
    return rmse


def train(params):
    print(params)

    dataset = MovieLens(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.review_feat_size,
                        params.num_factor,
                        symm=params.gcn_agg_norm_symm)
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.movie_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    net = Net(params, dataset.num_user, dataset.num_movie, dataset.review_data_dict, dataset.num_rating, dataset.rating_split)
    net = net.to(params.device)

    nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(params.device)
    rating_loss_net = nn.CrossEntropyLoss() if params.train_classification else nn.MSELoss()
    learning_rate = params.train_lr
    optimizer = get_optimizer(params.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    # prepare training data
    if params.train_classification:
        train_gt_labels = dataset.train_labels
        train_gt_ratings = dataset.train_truths
    else:
        train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

    # declare the loss information
    best_valid_rmse = np.inf
    best_test_rmse = np.inf
    no_better_valid = 0
    best_iter = -1

    for key in dataset.review_data_dict.keys():
        dataset.review_data_dict[key] = dataset.review_data_dict[key].to(params.device)
    for i, graph in enumerate(dataset.train_enc_graphs):
        dataset.train_enc_graphs[i] = dataset.train_enc_graphs[i].int().to(params.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.valid_enc_graphs):
        dataset.valid_enc_graphs[i] = dataset.valid_enc_graphs[i].int().to(params.device)
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.test_enc_graphs):
        dataset.test_enc_graphs[i] = dataset.test_enc_graphs[i].int().to(params.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)
    for i, graph in enumerate(dataset.test_dec_subgraphs):
        dataset.test_dec_subgraphs[i] = dataset.test_dec_subgraphs[i].int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        pred_ratings1, h_fea1, int_dists1, user1, item1, user_out1, movie_out1 = net(dataset.train_enc_graphs, dataset.train_dec_graph, dataset.review_data_dict)
        pred_ratings2, h_fea2, int_dists2, user2, item2, user_out2, movie_out2 = net(dataset.train_enc_graphs, dataset.train_dec_graph, dataset.review_data_dict)
        loss_cl = cal_c_loss(h_fea1, h_fea2, int_dists1, dataset.rating_split, params.num_pos)
        r_loss = (rating_loss_net(pred_ratings1, train_gt_labels).mean() + rating_loss_net(pred_ratings2, train_gt_labels).mean())/2
        loss = r_loss + loss_cl*params.lamda

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        if params.train_classification:
            real_pred_ratings = (th.softmax(pred_ratings1, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        else:
            real_pred_ratings = pred_ratings1

        train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()

        valid_rmse = evaluate(args=params, net=net, dataset=dataset, segment='valid')
        logging_str = f"Iter={iter_idx:>3d}, " \
                      f"Train_RMSE={train_rmse:.4f}, Valid_RMSE={valid_rmse:.4f},  Train_loss={loss:.4f}, "

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            no_better_valid = 0
            best_iter = iter_idx
            test_rmse = evaluate(args=params, net=net, dataset=dataset, segment='test')

            test_rmse = test_rmse * test_rmse
            best_test_rmse = test_rmse
            logging_str += 'Test RMSE={:.4f}'.format(test_rmse)
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
                print("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    print("\tChange the LR to %g" % new_lr)
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        print(logging_str)
    print(f'Best Iter Idx={best_iter}, Best Valid RMSE={best_valid_rmse:.4f}, Best Test RMSE={best_test_rmse:.4f}')
    print(params.model_save_path)


if __name__ == '__main__':
    config_args = config()
    train(config_args)