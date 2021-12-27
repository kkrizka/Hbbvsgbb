import sonnet as snt

class NormedMLP(snt.Module):
    """
    Wrapper around `snt.nets.MLP` that adds a `LayerNorm` layer for the inputs.
    """
    def __init__(self,output_sizes):
        super(NormedMLP, self).__init__()
        self.norm = snt.LayerNorm(-1, create_scale=True, create_offset=True)
        self.mlp = snt.nets.MLP(output_sizes)
    def __call__(self, input):
        output = self.norm(input)
        return self.mlp(output)