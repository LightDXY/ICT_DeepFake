import torch
from torch import optim
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torchvision import transforms as trans
import os
import argparse
import math
import time
import bcolz
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms import ToPILImage
from sklearn import metrics
from verifacation import calculate_roc_ex, evaluate, evaluate_new
import random

from knn_cuda import KNN
import torch.distributed as dist
from collections import OrderedDict
from torch.autograd import Variable

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import models.ict as ict

torch.backends.cudnn.enabled = True



def print_rank(*strs):
    if dist.get_rank() == 0:
        print (strs)
      
class face_learner(object):
    def __init__(self, conf, inference=False):
        #print(conf)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        print ('ICT EVALUATE')
        
        conf.embedding_size = 384
        
        if conf.net_mode == 'ict_base':
            print ('ICT Base Model')
            self.model = ict.combface_base_patch8_112()
        else:
            print ('Error Model:', conf.net_mode)
            exit(0)

        print_rank('Backbone {} model generated'.format(conf.net_mode))
    
        if not inference:
            self.eval_transform = trans.Compose([
                trans.Resize((112, 112)),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            self.step = -1
            self.start_epoch = 0
            
            best_path = os.path.join(conf.model_path, 'model_latest.pth')
            if os.path.exists(best_path):
                load_path = conf.model_path
            else:
                load_path = None
            if ( not load_path is None) and self.rank == 0:
                self.start_epoch = self.load_state(conf, load_path, 'latest.pth', model_only=True)
                print ('Load from {0}, epoch {1}'.format(load_path, self.start_epoch))
                self.start_epoch += 1

            self.model = self.model.to(conf.device)

            if self.world_size > 1:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[conf.device])

                #self.head = torch.nn.parallel.DistributedDataParallel(self.head, device_ids=[conf.device])
                start_epoch = torch.Tensor([self.start_epoch]).cuda()
                dist.barrier()
                dist.broadcast(start_epoch, 0)
                self.start_epoch = int(start_epoch.cpu().data.item())
            
            #self.bulid_all(conf)
            self.eval_all(conf)
            exit(0)

                    
    def eval_all(self, conf, per='None'):
        if conf.aug_test:
            pers = []
            pers.extend(['None'])
            pers.extend(['CS#0.0', 'CS#0.1', 'CS#0.2', 'CS#0.3', 'CS#0.4'])
            pers.extend(['CC#0.35', 'CC#0.475', 'CC#0.6', 'CC#0.725', 'CC#0.85'])
            pers.extend(['BW#80', 'BW#64', 'BW#48', 'BW#32', 'BW#16'])
            pers.extend(['GNC#0.05', 'GNC#0.01', 'GNC#0.005', 'GNC#0.002', 'GNC#0.001'])
            pers.extend(['GB#21', 'GB#17', 'GB#13', 'GB#9', 'GB#7'])
            pers.extend(['JPEG#6', 'JPEG#5', 'JPEG#4', 'JPEG#3', 'JPEG#2'])
            pers.extend(['REAL_JPEG#90', 'REAL_JPEG#70', 'REAL_JPEG#50', 'REAL_JPEG#30', 'REAL_JPEG#20'])
        else:
            pers = ['None']

        query = self.load_ref(conf, 1)
        print ('Load Ref Finished:', query['inner'].shape)
        json_path = 'DATASET/paths/'
        for per in pers:
            print ("\n\nRank {0} Evaluating with {1}...".format(self.rank, per))
            from data.ict_test_data import McDataset as fakeset
            fakes = [
                     ['FF', 'ff_real', 'ff_fake'],
                     ]
            for fake in fakes:
                fake_type = fake[0]
                real_name = fake[1].replace('/', '_')
                fake_name = fake[2].replace('/', '_')
                eval_ds = fakeset(json_path, real_name, fake_name, perturb = per, transform = self.eval_transform)
                sampler = None
                loader = DataLoader(eval_ds, batch_size=200, shuffle=(sampler is None), pin_memory=True, num_workers=8, sampler = sampler)
                self.evaluate(conf, loader, real_name, fake_name, query, per = per)
                del eval_ds
                del loader

    def load_ref(self, conf, drop):
        query = {}
        files = os.listdir(conf.model_path)
        temp_inner = []
        temp_outer = []
        for fi in files:
            if 'ref.pkl' == fi:
                temp_query = torch.load(os.path.join(conf.model_path, fi))
                print ('Loading:', fi, 'Drop:', 1-drop)
                bz = temp_query['inner'].shape[0]
                idx_shuffle = torch.randperm(bz)[:int(bz * drop)]
                temp_inner.append(temp_query['inner'][idx_shuffle])
                temp_outer.append(temp_query['outer'][idx_shuffle])
        if len(temp_inner) == 0:
            print ('No reference set found, ICT-Ref can not work.')
            exit(0)
        query['inner'] = torch.cat(temp_inner, 0).cuda().unsqueeze(0)
        query['outer'] = torch.cat(temp_outer, 0).cuda().unsqueeze(0)
        return query


    def evaluate(self, conf, loader, real_name, name, query, nrof_folds = 5, tta = False, per = ''):
        self.model.eval()
        idx = 0
        
        embeddings1 = []
        embeddings2 = []
        q_embeddings1 = []
        q_embeddings2 = []
        r_embeddings1 = []
        r_embeddings2 = []
        issame = []
        count = 0
        tot = min(500, len(loader))
        tri_num = 1
        self.knn = KNN(k=tri_num, transpose_mode=True)
        with torch.no_grad():
            for imgs, labels in iter(loader):
                #print (imgs.size())
                imgs = imgs.cuda()
                labels = labels.cuda()
                
                inner_emb, outer_emb = self.model(imgs)
                embeddings1.extend(inner_emb.cpu().numpy())
                embeddings2.extend(outer_emb.cpu().numpy())

                _, idx = self.knn(query['inner'], inner_emb.unsqueeze(0))
                tars = query['outer'][0][idx[0,:,0]]
                for i in range(1, tri_num):
                    tars += query['outer'][0][idx[0,:,i]]
                tars = tars / tri_num
                q_embeddings2.extend(tars.cpu().numpy())
                tars = query['inner'][0][idx[0,:,0]]
                for i in range(1, tri_num):
                    tars += query['inner'][0][idx[0,:,i]]
                tars = tars / tri_num
                r_embeddings1.extend(tars.cpu().numpy())

                _, idx = self.knn(query['outer'], outer_emb.unsqueeze(0))
                tars = query['inner'][0][idx[0,:,0]]
                for i in range(1, tri_num):
                    tars += query['inner'][0][idx[0,:,i]]
                tars = tars / tri_num
                q_embeddings1.extend(tars.cpu().numpy())
                tars = query['outer'][0][idx[0,:,0]]
                for i in range(1, tri_num):
                    tars += query['outer'][0][idx[0,:,i]]
                tars = tars / tri_num
                r_embeddings2.extend(tars.cpu().numpy())

                temp = [True if labels[i] else False for i in range(len(labels))]
                issame.extend(temp)
                count += 1
                if count % 100 == 0:
                    print_rank(f"{str(count)}/{str(tot)}")
                if count == tot:
                    break
            
        embeddings1 = np.asarray(embeddings1)
        embeddings2 = np.asarray(embeddings2)

        q_embeddings1 = np.asarray(q_embeddings1)
        q_embeddings2 = np.asarray(q_embeddings2)
        r_embeddings1 = np.asarray(r_embeddings1)
        r_embeddings2 = np.asarray(r_embeddings2)
        issame = np.asarray(issame)
        

        thres = 0.5
        thres2 = 0.5
        #print ('THRES:', thres, thres2)
        
        tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings1, embeddings2, issame, nrof_folds)
        auc = metrics.auc(fpr, tpr)
        for temp_rank in range(self.world_size):
            #dist.barrier()
            if self.rank == temp_rank:
                print ("Rank{4} Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'ICT', self.rank))

        if self.rank != 0:
            return
        
        dist1 = np.sum(np.square(np.subtract(embeddings1, embeddings2)), 1)
        dist2 = np.sum(np.square(np.subtract(embeddings2, q_embeddings2)), 1)
        dist3 = np.sum(np.square(np.subtract(embeddings1, q_embeddings1)), 1)

        tau = 0.5
        dis_exp2 = np.sum(np.square(np.subtract(embeddings1, r_embeddings1)), 1)
        #print ('DIS2', dis_exp2.mean())
        dis_exp2 = 0.75/(1+np.exp((dis_exp2 - thres)/tau))
        

        dis_exp3 = np.sum(np.square(np.subtract(embeddings2, r_embeddings2)), 1)
        #print ('DIS3', dis_exp3.mean())
        dis_exp3 = 0.75/(1+np.exp((dis_exp3 - thres2)/tau))
        
        dis_exp1 = 2 - dis_exp2 - dis_exp3
        #print (dis_exp1.mean(), dis_exp2.mean(), dis_exp3.mean())

        all_dist = dist1*dis_exp1 + dist2*dis_exp2 + dist3*dis_exp3
        thresholds = np.arange(0, 10, 0.01)
        tpr, fpr, accuracy, best_thresholds = calculate_roc_ex(thresholds, all_dist, issame)
        auc = metrics.auc(fpr, tpr)

        print ("Rank{4} Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'ICT_Ref', self.rank))

        tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings1, q_embeddings1, issame, nrof_folds)
        auc = metrics.auc(fpr, tpr)
        print ("Rank{4} Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'inner and query inner', self.rank))
 
        tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings2, q_embeddings2, issame, nrof_folds)
        auc = metrics.auc(fpr, tpr)
        print ("Rank{4} Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'outer and query outer', self.rank))

 
    def load_state(self, conf, save_path, fixed_str, model_only=False):
        save_dic = torch.load( os.path.join(save_path, 'model_{}'.format(fixed_str)), map_location='cpu')
        epoch = save_dic['epoch']
        
        self.model = self.model.cpu()
        self.model.load_state_dict(save_dic['model'])
        self.model = self.model.cuda(conf.device)

        if not model_only:
            self.head = self.head.cpu()
            self.head.load_state_dict(torch.load(os.path.join(save_path, 'head_{}'.format(fixed_str)), map_location='cpu'))
            self.head = self.head.cuda(conf.device)
        save_dic = ''
        new_state_dict = ''
        del save_dic
        del new_state_dict
        torch.cuda.empty_cache()
        return epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for ICT DeepFake detection')

    parser.add_argument("--net_mode", default='ict_base', type=str)
    parser.add_argument("--aug_test", action='store_true', help='test with perturped input')
    parser.add_argument("-name", "--dump_name", default='mask_test', type=str)
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = world_size > 1
    print("world size is {}".format(world_size))
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        port = '235'+str(int (np.floor(random.random()*100)))
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:'+port, rank=0, world_size=1)
    args.device = args.local_rank

    print (f"Init finished, World_size {str(dist.get_world_size())}, Rank{str(dist.get_rank())}")

    args.model_path = os.path.join('./PRETRAIN/', args.dump_name)
    
    learner = face_learner(args)

