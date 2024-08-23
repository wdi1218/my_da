import torch
import torch.nn as nn



class CrsAtten(nn.Module):
    def __init__(self):
        super(CrsAtten, self).__init__()
        # self.linear1 = nn.Linear(args.encoder_dim + args.decoder_dim, 1)
        self.linear1 = nn.Linear(128 + 128, 1)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_h):
        # encoder_out b*l*d1       decoder_h   b*d2
        # c_t   b*d1
        l = encoder_out.size(1)
        decoder_h = decoder_h.unsqueeze(1).repeat(1,l,1)
        cat_h = torch.cat([encoder_out, decoder_h], -1)
        cat_h = self.linear1(cat_h).squeeze()
        alpha = self.softmax1(cat_h)

        alpha = alpha.unsqueeze(-1).permute(0,2,1)  # b*1*l
        ctx = torch.bmm(alpha, encoder_out).squeeze()     # b*d1

        return ctx




