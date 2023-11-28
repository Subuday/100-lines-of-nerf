import torch


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
