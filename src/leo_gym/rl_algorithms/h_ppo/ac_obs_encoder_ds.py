import torch as T
import torch.nn as nn
from torch_geometric.nn.aggr.deep_sets import DeepSetsAggregation

class ObservationEncoder(nn.Module):
    def __init__(
        self, 
        obs_dim: None,
        net_arch=[64, 64, nn.Tanh], 
        device: str = "cpu",
        obs_sat_size:int=4,
        obs_deb_size:int=6,
    ):
        """
        Observation encoder for H-PPO that supports variable inputs with deepsets layer
        
        :param obs_sat_size: Satellite observations states
        :type obs_sat_size: int
        :param obs_deb_size: Debris observation states
        :type obs_deb_size: int
        :param net_arch: network architecture number of layers with neuron size and activation function
        :type obs_deb_size: list
        """
        super().__init__()    
            
        self.device = T.device(device)

        if isinstance(net_arch, int):
            hidden_sizes, activation_cls = [net_arch], nn.Tanh
        elif isinstance(net_arch[-1], type) and issubclass(net_arch[-1], nn.Module):
            hidden_sizes, activation_cls = list(net_arch[:-1]), net_arch[-1]
        else:
            hidden_sizes, activation_cls = list(net_arch), nn.Tanh

        gain = nn.init.calculate_gain("tanh")
        
        self.obs_sat_size = obs_sat_size
        self.obs_deb_size = obs_deb_size

        nds_linear = nn.Linear(self.obs_sat_size, hidden_sizes[0])
        nn.init.orthogonal_(nds_linear.weight, gain=gain)
        nds_linear.bias.data.zero_()
        self.fc_nds = nn.Sequential(nds_linear, activation_cls())

        ds_local = nn.Linear(self.obs_deb_size, hidden_sizes[0])
        nn.init.orthogonal_(ds_local.weight, gain=gain)
        ds_local.bias.data.zero_()
        local_nn = nn.Sequential(ds_local, activation_cls())

        ds_global = nn.Linear(hidden_sizes[0], hidden_sizes[0])
        nn.init.orthogonal_(ds_global.weight, gain=gain)
        ds_global.bias.data.zero_()
        global_nn = nn.Sequential(ds_global, activation_cls())

        self.ds = DeepSetsAggregation(local_nn=local_nn, global_nn=global_nn)

        layers, in_dim = [], 2 * hidden_sizes[0]
        for h in hidden_sizes[1:]:
            linear = nn.Linear(in_dim, h)
            nn.init.orthogonal_(linear.weight, gain=gain)
            linear.bias.data.zero_()
            layers += [linear, activation_cls()]
            in_dim = h

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = hidden_sizes[-1]
        self.to(self.device)

    def forward(self, obs: T.Tensor) -> T.Tensor:
        """
        Encode satellite and debris observations into a latent representation.

        Zero-padded debris observations are ignored during DeepSets aggregation.

        :param obs: Observation tensor of shape (..., obs_dim).
        :type obs: torch.Tensor

        :return: Encoded latent representation.
        :rtype: torch.Tensor
        """

        obs = T.as_tensor(obs, dtype=T.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        leading_shape = obs.shape[:-1]

        nds_obs = obs[..., :self.obs_sat_size]
        ds_obs = obs[..., self.obs_sat_size:].view(
            *leading_shape, -1, self.obs_deb_size
        )

        nds_latent = self.fc_nds(nds_obs)

        # Flatten leading dimensions
        B = ds_obs.reshape(-1, ds_obs.shape[-2], ds_obs.shape[-1]).shape[0]
        N = ds_obs.shape[-2]
        D = ds_obs.shape[-1]

        ds_obs = ds_obs.reshape(B, N, D)
        nds_latent = nds_latent.reshape(B, -1)

        # Ignore zero-padded debris
        mask = ds_obs.abs().sum(dim=-1) > 0

        valid = ds_obs[mask]
        batch = (
            T.arange(B, device=self.device)
            .unsqueeze(1)
            .expand(B, N)[mask]
        )

        # Add one dummy zero debris for environments with no valid debris
        counts = T.bincount(batch, minlength=B)
        empty = (counts == 0).nonzero(as_tuple=False).flatten()

        if len(empty):
            valid = T.cat(
                (valid, T.zeros(len(empty), D, device=self.device)),
                dim=0,
            )
            batch = T.cat((batch, empty), dim=0)

        ds_latent = self.ds(valid, index=batch)

        # Restore original leading dimensions
        ds_latent = ds_latent.view(*leading_shape, -1)
        nds_latent = nds_latent.view(*leading_shape, -1)

        x = T.cat((nds_latent, ds_latent), dim=-1)

        return self.trunk(x)