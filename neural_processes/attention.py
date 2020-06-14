import functools

import torch


class Uniform:
    def forward(self, context_x, r, target_x=None):
        if target_x is None:
            return r.mean(dim=1)
        else:
            return r.mean(dim=1).unsqueeze(dim=1).repeat([1, target_x.size()[1], 1])


class BasicAttention(torch.nn.Module):
    def __init__(self, key_mlp, attention_func, **kwargs):
        super().__init__()
        self.key_mlp = key_mlp
        self.attention_func = attention_func
        self.kwargs = kwargs

    def forward(self, context_x, r, target_x):
        keys = self.key_mlp.forward(context_x)
        query = self.key_mlp.forward(target_x)
        return self.attention_func(keys, r, query, **self.kwargs)


def laplace(keys, values, query):
    W = (query.unsqueeze(1) - keys.unsqueeze(2)).sum(dim=-1).softmax(dim=-1)
    return torch.matmul(W.transpose(1, 2), values)


def dot_product(keys, values, query):
    dotted = torch.softmax(torch.matmul(query, keys.transpose(1, 2)) / query.size()[2], dim=-1)
    return torch.matmul(dotted, values)


Laplace = functools.partial(BasicAttention.__init__, attention_func=laplace)
DotProduct = functools.partial(BasicAttention, attention_func=dot_product)


class Multihead(torch.nn.Module):
    def __init__(self, key_mlp, representation_size, num_heads):
        super().__init__()
        self.key_mlp = key_mlp
        self.num_heads = num_heads
        key_size = key_mlp.final.out_features
        if representation_size % num_heads != 0:
            raise ValueError(
                f"Number of heads {num_heads} should divide"
                f" representation size {representation_size}"
            )
        intermediate_size = representation_size // num_heads
        self.w_k = torch.nn.ModuleList(
            [torch.nn.Linear(key_size, intermediate_size, bias=False) for _ in range(num_heads)]
        )
        self.w_v = torch.nn.ModuleList(
            [torch.nn.Linear(key_size, intermediate_size, bias=False) for _ in range(num_heads)]
        )
        self.w_q = torch.nn.ModuleList(
            [
                torch.nn.Linear(representation_size, intermediate_size, bias=False)
                for _ in range(num_heads)
            ]
        )
        self.w_out = torch.nn.Linear(representation_size, representation_size, bias=False)

    def forward(self, context_x, r, target_x):
        keys = self.key_mlp.forward(context_x)
        query = self.key_mlp.forward(target_x)
        return self.multihead(keys, r, query)

    def multihead(self, keys, values, query):
        out = [self._single_head(i, keys, values, query) for i in range(self.num_heads)]
        return self.w_out.forward(torch.cat(out, dim=-1))

    def _single_head(self, i, keys, values, query):
        return dot_product(
            self.w_k[i].forward(keys), self.w_v[i].forward(values), self.w_q[i].forward(query)
        )
