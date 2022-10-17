import torch
import torch.nn as nn
from easydict import EasyDict

from network.ibrnet import IBRNetWithNeuRay
from network.semantic_nerf import *

def get_dir_diff(prj_dir,que_dir):
    rfn, qn, rn, dn, _ = prj_dir.shape
    dir_diff = prj_dir - que_dir.unsqueeze(0)  # rfn,qn,rn,dn,3
    dir_dot = torch.sum(prj_dir * que_dir.unsqueeze(0), -1, keepdim=True)
    dir_diff = torch.cat([dir_diff, dir_dot], -1)  # rfn,qn,rn,dn,4
    dir_diff = dir_diff.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
    return dir_diff

class DefaultAggregationNet(nn.Module):
    default_cfg={
        'sample_num': 64,
        'neuray_dim': 32,
        'use_img_feats': False,
        'enable_semantic':True,
        'use_sem_nerf': False,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}
        # args = EasyDict(anti_alias_pooling=False,local_rank=0)
        dim = self.cfg['neuray_dim']
        if not self.cfg['use_sem_nerf']:
            self.agg_impl = IBRNetWithNeuRay(dim,n_samples=self.cfg['sample_num'])
        
        else:
            #TODO
            embed_fn, input_ch = get_embedder(10, i=0, scalar_factor=10)

            input_ch_views = 0
            embeddirs_fn = None
            embeddirs_fn, input_ch_views = get_embedder(4, i=0, scalar_factor=1)
         
            self.agg_impl = Semantic_NeRF(enable_semantic=True, num_semantic_classes=40, D=8, W=256,
                input_ch=input_ch, input_ch_views=input_ch_views, input_ch_feat=32, skips=[4], use_viewdirs=True)
            self.embed_fn, self.embeddirs_fn = embed_fn, embeddirs_fn
            # ENDTODO

        self.prob_embed = nn.Sequential(
            nn.Linear(2+32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, prj_dict, que_dir):
        """
        :param prj_dict
             prj_ray_feats: rfn,qn,rn,dn,f
             prj_hit_prob:  rfn,qn,rn,dn,1
             prj_vis:       rfn,qn,rn,dn,1
             prj_alpha:     rfn,qn,rn,dn,1
             prj_rgb:       rfn,qn,rn,dn,3
             prj_dir:       rfn,qn,rn,dn,3
        :param que_dir:       qn,rn,dn,3
        :return: qn,rn,dn
        """
        hit_prob_val = (prj_dict['hit_prob']-0.5)*2
        vis_val = (prj_dict['vis']-0.5)*2

        prj_hit_prob, prj_vis, prj_rgb, prj_dir, prj_ray_feats, prj_pts = \
            hit_prob_val, vis_val, prj_dict['rgb'], prj_dict['dir'], prj_dict['ray_feats'], prj_dict['pts_3d']
        rfn,qn,rn,dn,_ = hit_prob_val.shape

        prob_embedding = self.prob_embed(torch.cat([prj_ray_feats, prj_hit_prob, prj_vis],-1))

        dir_diff = get_dir_diff(prj_dir, que_dir)
        valid_mask = prj_dict['mask']
        valid_mask = valid_mask.float() # rfn,qn,rn,dn
        valid_mask = valid_mask.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)

        prj_img_feats = prj_dict['img_feats']
        prj_img_feats = torch.cat([prj_rgb, prj_img_feats], -1)
        prj_img_feats = prj_img_feats.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
        prob_embedding = prob_embedding.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
        
        if not self.cfg['use_sem_nerf']:
            outs = self.agg_impl(prj_img_feats, prob_embedding, dir_diff, valid_mask)

        else:
            # use semantic_nerf instead of ibrnet
            embed_pts  = np.squeeze(self.embed_fn(prj_pts)).reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3) 
            embed_dirs = np.squeeze(self.embeddirs_fn(prj_dir)).reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
            input_comb = torch.cat([embed_pts, 
                                    embed_dirs, 
                                    prob_embedding], -1)
            outs = self.agg_impl(input_comb)
        
        
        colors = outs[...,:3] # qn*rn,dn,3
        density = outs[...,3] # qn*rn,dn,0

        if not self.cfg['use_sem_nerf']:
            sem_logits=None
            #TODO
            return density.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3), sem_logits
        else:
            sem_logits = None
            if(self.cfg['enable_semantic']):
                sem_logits = outs[... , 4:].reshape(qn, rn, dn, -1)
            return density.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3), sem_logits


name2agg_net={
    'default': DefaultAggregationNet
}