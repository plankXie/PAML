import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


class VisualDict(nn.Module):
    def __init__(self, num_tokens, token_dim, decay=0.1, max_decay=0.99, eps=1e-5) -> None:
        super().__init__()
        self.num_tokens = num_tokens #2048
        self.token_dim = token_dim   #768
        self.decay = decay           #0.4
        self.cur_decay = decay       #0.4
        self.max_decay = max_decay   #0.99
        self.eps = eps               #1e-05
        self.topk = 5
        self.temperature = 1.0

        embed = torch.randn(num_tokens, token_dim)  #初始化对应的原型库
        self.register_buffer("embed", embed)        ##将原型库注册为buffer,因而不会被当作模型参数进行优化
        nn.init.normal_(self.embed)                 #原型库正态分布初始化
        # embed = torch.normal(num_tokens, token_dim)
        # self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_tokens)) #初始化存储每个token聚类大小的buffer
        self.register_buffer("cluster_sum", torch.zeros(num_tokens))  #初始化存储每个token聚类和的buffer
        self.register_buffer("embed_avg", torch.zeros(num_tokens, token_dim)) #初始化存储每个token嵌入平均值的buffer

    def set_decay_updates(self, num_update) -> None:
        self.cur_decay = min(self.cur_decay * num_update, self.max_decay)


    def forward(self, inputs_flatten: torch.Tensor):
        # 计算输入与所有原型token的距离
        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embed.data ** 2, dim=1) -
                     2 * torch.matmul(inputs_flatten, self.embed.data.t()))

        # 计算soft权重
        if self.topk is not None:
            # 如果设置了topk，只考虑距离最近的k个token
            topk_distances, topk_indices = torch.topk(-distances, k=self.topk, dim=1)
            topk_distances = -topk_distances

            # 计算softmax权重
            weights = torch.softmax(-topk_distances / self.temperature, dim=1)

            # 创建稀疏的权重矩阵
            encodings = torch.zeros(distances.shape, device=inputs_flatten.device)
            encodings.scatter_(1, topk_indices, weights)
        else:
            # 否则考虑所有token，使用softmax
            encodings = torch.softmax(-distances / self.temperature, dim=1)

        encoding_indices = torch.argmax(encodings, dim=1).unsqueeze(1)  # 仍然保留最可能的token索引

        if self.training:
            # 更新统计量时使用soft权重
            encoding_sum = encodings.sum(dim=0)
            sum_inplace(self.cluster_sum, encoding_sum)
            ema_tensor_inplace(self.cluster_size, encoding_sum, self.cur_decay)

            embed_sum = torch.matmul(encodings.t(), inputs_flatten)
            ema_tensor_inplace(self.embed_avg, embed_sum, self.cur_decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.num_tokens, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        # 使用soft权重进行量化
        quantize = torch.matmul(encodings, self.embed)
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten

        return quantize, encoding_indices




