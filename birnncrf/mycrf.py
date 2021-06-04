import torch.nn as nn
import torch


IMPOSSIBLE = -1e4

def log_sum_exp(x):
    max_score = x.max(-1)[0]
    return max_score+(x-max_score.unsqueeze(-1)).exp().sum(-1).log()

class crf(nn.Module):

    def __init__(self,in_feartures,num_tags):
        super(crf,self).__init__()

        self.num_tags = num_tags+2
        self.start_idx = self.num_tags-2
        self.stop_idx = self.num_tags-1

        self.fc = nn.Linear(in_feartures,self.num_tags)

        self.transitions = nn.Parameter(torch.randn(self.num_tags,self.num_tags),requires_grad=True)
        self.transitions.data[self.start_idx,:] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE


    def forward(self,features,masks):
        features = self.fc(features)
        return self.__viterbi_decode(features,masks[:,:features.size(1)].float())

    def loss(self,features,ys,masks):

        features = self.fc(features)
        L = features.size(1)
        masks_ = masks[:,:L].float()
        forward_score = self.__forward_algorithm(features,masks_)
        gold_score = self.__score_sentence(features,ys[:,:L].long(),masks_)
        loss = (forward_score-gold_score).mean()
        return loss

    def __score_sentence(self,features,tags,masks):
        B,L,C = features.shape
        emit_scores = features.gather(dim=2,index=tags.unsqueeze(-1)).squeeze(-1)

        start_tag = torch.full((B,1),self.start_idx,dtype=torch.long,device=tags.device)
        tags = torch.cat([start_tag,tags],dim=1)
        trans_scores = self.transitions[tags[:,1:],tags[:,:-1]]

        last_tag = tags.gather(dim=1,index=masks.sum(1).long().unsqueeze(1)).squeeze(1)
        last_score = self.transitions[self.stop_idx,last_tag]
        score = ((trans_scores+emit_scores)*masks).sum(1)+last_score

        return score

    def __viterbi_decode(self,features, masks):

        B,L,C = features.shape

        bps = torch.zeros(B,L,C,dtype=torch.long,device=features.device)

        max_score = torch.full((B,C),IMPOSSIBLE,device=features.device)

        max_score[:,self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:,t].unsqueeze(1)
            emit_score_t = features[:,t]

            acc_score_t = max_score.unsqueeze(1)+self.transitions
            acc_score_t,bps = acc_score_t.max(dim=-1)
            acc_score_t+=emit_score_t
            max_score = acc_score_t*mask_t + max_score * (1-mask_t)

        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        best_paths = []
        bps = bps.cpu().numpy()

        for b in range(B):

            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b,:].sum().item())
            best_path = [best_tag_b]
            for bps_t in reversed(bps[b,:seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            best_paths.append(best_path[-2::-1])

        return best_score,best_paths


    def __forward_algorithm(self,features,masks):

        B,L,C = features.shape
        scores = torch.full((B,C),IMPOSSIBLE,device=features.device)
        scores[:,self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)

        for t in range(L):
            emit_score_t = features[:,t].unsqueeze(2)
            score_t = scores.unsqueeze(1)+trans+emit_score_t
            score_t = log_sum_exp(score_t)

            mask_t = masks[:,t].unsqueeze(1)
            scores = score_t * mask_t+scores*(1-mask_t)

        scores =log_sum_exp(scores+self.transitions[self.stop_idx])

        return scores
