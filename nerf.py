import torch
from torch import nn
import torch.nn.functional as F


class Embedder:
    def __init__(self):
        self.functions, self.output_dim = self.create_embedding_functions()

    def create_embedding_functions(self):
        functions = []
        input_dim = 3
        output_dim = 0

        functions.append(lambda x: x)
        output_dim += input_dim

        positional_frequencies = 2. ** torch.linspace(0., 3, steps=4)

        for freq in positional_frequencies:
            for p_fn in [torch.sin, torch.cos]:
                functions.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                output_dim += input_dim

        return functions, output_dim

    def embedding(self, inputs):
        return torch.cat([fn(inputs) for fn in self.functions], -1)


class NeRF(nn.Module):
    def __init__(self, input_ch_pts, input_ch_views):
        super(NeRF, self).__init__()
        self.d = 8
        self.w = 256
        self.input_ch_pts = input_ch_pts
        self.input_ch_views = input_ch_views
        self.skips = [4]

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_pts, self.w)] +
            [nn.Linear(self.w, self.w) if i not in self.skips else nn.Linear(self.w + self.input_ch, self.w) for i in range(self.d - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + self.w, self.w // 2)])

        self.feature_linear = nn.Linear(self.w, self.w)
        self.alpha_linear = nn.Linear(self.w, 1)
        self.rgb_linear = nn.Linear(self.w // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch_pts, self.input_ch_views], dim=-1)
        h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs
