import torch
import torch.nn as nn
from compressor.ckbd import *
from compressor.layers import *
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, in_nc, M):
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlock(in_nc, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlockWithStride(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            conv3x3(M, M)
        )

    def forward(self, x):
        return self.g_a(x)


class Decoder(nn.Module):
    def __init__(self, M):
        super().__init__()

        self.g_s = nn.Sequential(
            conv3x3(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M),
            ResidualBlock(M, M)
        )

    def forward(self, x):
        return self.g_s(x)

class HyperEncoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        self.hyper_enc = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N),
            ResidualBlockWithStride(N, N),
        )

    def forward(self, x):
        return self.hyper_enc(x)


class HyperDecoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(N, M),
            ResidualBlockUpsample(M, M),
            ResidualBlock(M, M * 3 // 2),
            ResidualBlock(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        return self.hyper_dec(x)


class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        return self.fushion(channel_params)


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        return self.fusion(params)


class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """

    def __init__(self, num_embed, embed_dim, beta=0.25, distance='l2',
                 anchor='closest', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))
        self.register_buffer('usage', torch.zeros(self.num_embed, dtype=torch.int), persistent=False)

    def reset_usage(self):
        self.usage = self.usage * 0

    def get_usage(self):
        codebook_usage = 1.0 * (self.num_embed - (self.usage == 0).sum()) / self.num_embed
        return codebook_usage

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
            torch.sum(self.embedding.weight ** 2, dim=1) + \
            2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)

        loss = 0.0
        perplexity = None

        if self.training:
            # compute loss for embedding
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
            # preserve gradients, STE
            z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        b, _, h, w = z_q.shape
        encoding_indices = rearrange(encoding_indices, '(b h w) -> b h w', b=b, h=h, w=w).contiguous()

        min_encodings = encodings

        if not self.training:
            for idx in range(self.num_embed):
                self.usage[idx] += (encoding_indices == idx).sum()

        # online clustered reinitialisation for unoptimized points
        if self.training:
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(
                    1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True

            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0,
                                                                                                       keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)

    def quant(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', self.embedding.weight))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding.weight[min_encoding_indices].view(z.shape)

        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        b, _, h, w = z_q.shape
        encoding_indices = rearrange(min_encoding_indices, '(b h w) -> b h w', b=b, h=h, w=w).contiguous()

        return z_q, encoding_indices

    def get_codebook_entry(self, indices):
        b, h, w = indices.shape
        indices = indices.flatten().to(self.embedding.weight.device)
        z_q = self.embedding(indices).view(b, h, w, -1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """

    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(
                    0) > self.pool_size:  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features
