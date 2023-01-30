from typing import Tuple

import torch
import torch.nn as nn
import math
from models.ode import ODE_Pendulum, ODE_2ObjectsSpring, ODE_SlidingBlock, ODE_ThrownObject
from util.util import rotmat_2d, get_pixel_coords, applyHomography, interp1D

from torchdiffeq import odeint, odeint_adjoint


class Scene(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.background_representation = Representation_FourierFeatures(**kwargs)

        self.local_representation = None
        self.n_tsteps = None
        self.has_2ObjectSpring = False

        self.homography_matrix = nn.Parameter(torch.eye(3), requires_grad=True)
        self.use_homography = False

    def get_params(self):
        param_dict, n_parameters = self.background_representation.get_params()
        params = {'parameter_background': param_dict}

        if self.local_representation is not None:
            local_param_dict, n_local_parameters = self.local_representation.get_params()
            n_parameters += n_local_parameters
            params['parameter_obj'] = local_param_dict

        # Check if all parameters have been added
        n_params = 0
        for param in self.parameters():
            n_params += 1
        # -1 because of the homography matrix
        if n_params - 1 != n_parameters:
            print("WARNING: Not all parameters have been added!")

        return params

    def add_pendulum(
        self,
        A=0.25*torch.ones(2),
        c=torch.tensor(1.),
        l_pendulum=torch.tensor(1.),
        x0=torch.rand(2)-0.5,
        use_damping: bool = True,
        **kwargs
    ):
        self.local_representation = LocalRepresentation_Pendulum(
            A=A,
            c=c,
            l_pendulum=l_pendulum,
            x0=x0,
            use_damping=use_damping,
            **kwargs
        )

    def add_slidingBlock(
        self,
        alpha,
        p0,
        mu=torch.tensor(0.),
        **kwargs,
    ):
        self.local_representation = LocalRepresentation_SlidingBlock(
            mu=mu,
            alpha=alpha,
            p0=p0,
            **kwargs,
        )

    def add_thrownObject(
        self,
        p0,
        v0=torch.zeros(2),
        **kwargs
    ):
        self.local_representation = LocalRepresentation_ThrownObject(
            p0=p0,
            v0=v0,
            **kwargs
        )

    def add_2ObjectsSpring(
        self,
        **kwargs
    ):
        self.local_representation = LocalRepresentation_2ObjectsSpring(**kwargs)

        self.has_2ObjectSpring = True

    def add_posedLocal(
        self,
        t_poses,
        **kwargs
    ):
        self.local_representation = LocalRepresentation_posed(
            t_poses=t_poses,
            **kwargs
        )

    def update_trafo(self, tspan):
        self.local_representation.update_trafo(tspan)
        self.n_tsteps = len(tspan)

    def forward(self, x, only_background=False):
        if self.local_representation is None or only_background:
            colors = self.background_representation(x)

            return {
                'colors': colors
            }

        else:
            # Apply homography, if we use it
            if self.use_homography:
                x_for_local = applyHomography(self.homography_matrix, x)
            else:
                x_for_local = x
            # Evaluate local representation
            local_colors, mask = self.local_representation(x_for_local)

            # Evaluate global representation for all points
            background_colors = self.background_representation(x).unsqueeze(1)

            if self.has_2ObjectSpring:
                comb_mask = mask["mask_combined"]

                # Overlay the colors
                colors = comb_mask[..., None] * local_colors + (1-comb_mask[..., None]) * background_colors
            else:
                # Overlay the colors
                colors = mask[..., None] * local_colors + (1-mask[..., None]) * background_colors

            return {
                'colors': colors,
                'mask': mask,
                'colors_bg': background_colors,
                'colors_foreground': local_colors
            }

    def render_image(self, W, H, chunk_size=65536, only_background=False, normalize_by_H=False):
        is_training_initially = self.training
        self.eval()

        # Get number of time points
        if only_background:
            dim_time = []
        else:
            dim_time = [self.n_tsteps]

        # Get device of model
        device = next(self.parameters()).device

        # Create coordinates
        pixel_coords = get_pixel_coords(H, W, normalize_by_H, device)
        n_pixels = pixel_coords.shape[0]

        n_chunks = math.ceil(n_pixels / chunk_size)
        image = torch.zeros(n_pixels, *dim_time, 3, device=device)
        mask = torch.zeros(n_pixels, *dim_time, device=device)

        if self.has_2ObjectSpring:
            mask1 = torch.zeros(n_pixels, *dim_time, device=device)
            mask2 = torch.zeros(n_pixels, *dim_time, device=device)

        for i in range(n_chunks):
            start = i*chunk_size
            end = min(start+chunk_size, n_pixels)
            cur_inds = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=device
            )

            # Render current pixels
            with torch.no_grad():
                output = self.forward(pixel_coords[cur_inds, :], only_background)
                colors = output['colors']
                image[cur_inds, ...] = colors

                if not only_background:
                    cur_mask = output['mask']
                    if cur_mask is not None:
                        if self.has_2ObjectSpring:
                            mask[cur_inds, ...] = cur_mask['mask_combined']
                            mask1[cur_inds, ...] = cur_mask['mask1']
                            mask2[cur_inds, ...] = cur_mask['mask2']

                        else:
                            mask[cur_inds, ...] = cur_mask

        # Set back to training
        if is_training_initially:
            self.train()

        # Reshape image and return
        output = {}
        if only_background:
            output["Image"] = image.view(H, W, -1)
        else:
            output["Image"] = image.transpose(0, 1).view(*dim_time, H, W, -1)

        if not only_background:
            output["Mask"] = mask.transpose(0, 1).view(*dim_time, H, W)

            if self.has_2ObjectSpring:
                output['Mask1'] = mask1.transpose(0, 1).view(*dim_time, H, W)
                output['Mask2'] = mask2.transpose(0, 1).view(*dim_time, H, W)

        return output


class LocalRepresentation_OneObject(nn.Module):
    def __init__(
        self,
        use_adjoint: bool = True,
        **kwargs
    ):
        super().__init__()

        self.use_adjoint = use_adjoint

        # Transformations from and to the local coordinate system
        self.trafo_to_local = lambda x: x
        self.trafo_from_local = lambda x: x

        # Local Representation
        self.local_representation = Representation_FourierFeatures(output_dim=4, **kwargs)

    def get_params(self):
        params, n_params_repr = self.local_representation.get_params()

        params_physics = []
        names_params_physics = []

        for name, param in self.named_parameters():
            if "local_representation" not in name:
                params_physics += [param]
                names_params_physics += [name]

        params["params_physics"] = params_physics
        params["names_params_position"] = names_params_physics

        return params, n_params_repr + len(params_physics)

    def update_trafo(self, tspan):
        raise NotImplementedError("Update of trafo needs to be implemented for the specific physical system")

    def forward(self, x):
        # Transform input points to local coordinate system
        x_local = self.trafo_to_local(x)

        # Evaluate background representation
        output = self.local_representation(x_local)

        colors = output[..., :3]
        mask = output[..., 3]

        return colors, mask

    def render_local_object(self, W, H):
        # Get device of model
        device = next(self.parameters()).device

        # Create coordinates, normalized to the width
        H = 2*int(H/2)
        W = 2*int(W/2)
        xx = (torch.arange(W, dtype=torch.float32) - W/2) / (W/2)
        yy = (torch.arange(H, dtype=torch.float32) - H/2) / (H/2)
        x_grid, y_grid = torch.meshgrid(xx, yy)
        pixel_coords = torch.stack([x_grid.transpose(0, 1), y_grid.transpose(0, 1)], dim=2).view(-1, 2).to(device)

        out = self.local_representation(pixel_coords)
        mask = out[..., 3].view(H, W)
        colors = out[..., :3].view(H, W, -1)
        return mask, colors


class LocalRepresentation_Pendulum(LocalRepresentation_OneObject):
    def __init__(
        self,
        A=0.25*torch.ones(2),
        c=torch.tensor(1.),
        l_pendulum=torch.tensor(1.),
        x0=torch.randn(2),
        use_damping: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.A = nn.Parameter(A, requires_grad=True)
        self.x0 = nn.Parameter(x0, requires_grad=True)

        self.ode = ODE_Pendulum(c, l_pendulum, use_damping)

    def update_trafo(self, tspan):
        # Solve ODE for timespan
        if self.use_adjoint:
            y = odeint_adjoint(self.ode, self.x0, tspan)
        else:
            y = odeint(self.ode, self.x0, tspan)

        # Create rotation matrices from solution
        R = rotmat_2d(y[:, 0])

        # Update trafos
        self.trafo_to_local = lambda x: ((x - self.A) @ R).transpose(0, 1)
        self.trafo_from_local = lambda x: x @ R.transpose(1, 2) + self.A


class LocalRepresentation_SlidingBlock(LocalRepresentation_OneObject):
    def __init__(
        self,
        mu,
        alpha,
        p0,
        x0=torch.zeros(2),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ode = ODE_SlidingBlock(mu, alpha)
        self.p0 = nn.Parameter(p0, requires_grad=True)
        self.x0 = nn.Parameter(x0, requires_grad=True)

    def update_trafo(self, tspan):
        # Solve ODE for timespan
        if self.use_adjoint:
            y = odeint_adjoint(self.ode, self.x0, tspan)
        else:
            y = odeint(self.ode, self.x0, tspan)

        # Create rotation matrix
        R = rotmat_2d(self.ode.alpha)

        # Displacement Block
        d = torch.stack([y[:, 0], torch.zeros_like(y[:, 0])], dim=1)

        # Update trafos
        self.trafo_to_local = lambda x: ((x - self.p0) @ R.transpose(0, 1)).unsqueeze(1) - d.unsqueeze(0)
        self.trafo_from_local = lambda x: (x.unsqueeze(1) + d.unsqueeze(0)) @ R + self.p0


class LocalRepresentation_ThrownObject(LocalRepresentation_OneObject):
    def __init__(
        self,
        p0,
        v0=torch.zeros(2),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ode = ODE_ThrownObject()
        self.p0 = nn.Parameter(p0, requires_grad=True)
        self.v0 = nn.Parameter(v0, requires_grad=True)

    def update_trafo(self, tspan):
        # Solve ODE for timespan
        init_vals = torch.cat([torch.zeros_like(self.v0), self.v0])
        if self.use_adjoint:
            y = odeint_adjoint(self.ode, init_vals, tspan)
        else:
            y = odeint(self.ode, init_vals, tspan)

        # Trajectory object
        traj = y[:, :2]

        # Update trafos
        self.trafo_to_local = lambda x: (x - self.p0).unsqueeze(1) - traj.unsqueeze(0)
        self.trafo_from_local = lambda x: (x + self.p0).unsqueeze(1) + traj.unsqueeze(0)


class LocalRepresentation_posed(LocalRepresentation_OneObject):
    def __init__(
        self,
        t_poses,
        translations_init=None,
        angles_init=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.register_buffer('t_poses', t_poses)

        if translations_init is None:
            translations_init = torch.zeros(2, len(t_poses))
        if angles_init is None:
            angles_init = torch.zeros(1, len(t_poses))

        self.angles = nn.Parameter(angles_init, requires_grad=True)
        self.translations = nn.Parameter(translations_init, requires_grad=True)

    def update_trafo(self, tspan):
        angles = interp1D(tspan, self.t_poses, self.angles).squeeze()
        translations = interp1D(tspan, self.t_poses, self.translations)

        # Create rotation matrices from angles
        R = rotmat_2d(angles)

        # Hack to be able to train with sequence length 1
        if len(R.shape) < 3:
            R = R.unsqueeze(0)

        # Update trafos
        def trafo_to_local(x):
            diff = x.unsqueeze(0) - translations.transpose(0, 1).unsqueeze(1)

            result = torch.zeros_like(diff)

            for i in range(result.shape[0]):
                result[i] = diff[i] @ R[i]

            return result.transpose(0, 1)

        self.trafo_to_local = trafo_to_local


class LocalRepresentation_2ObjectsSpring(nn.Module):
    def __init__(
        self,
        p1_0=0.25*torch.ones(2),
        p2_0=0.75*torch.ones(2),
        v1_0=torch.zeros(2),
        v2_0=torch.zeros(2),
        k=torch.tensor(1.),
        eq_distance=torch.tensor(1.),
        use_adjoint: bool = True,
        **kwargs
    ):
        super().__init__()

        self.x0 = nn.Parameter(
            torch.cat([p1_0, p2_0, v1_0, v2_0]),
            requires_grad=True
        )

        self.ode = ODE_2ObjectsSpring(k, eq_distance)
        self.use_adjoint = use_adjoint

        # Transformations from and to the local coordinate system
        self.trafo_from_local1 = lambda x: x + p1_0
        self.trafo_to_local1 = lambda x: x - p1_0
        self.trafo_from_local2 = lambda x: x + p2_0
        self.trafo_to_local2 = lambda x: x - p2_0

        # Local Representation
        self.local_representation1 = Representation_FourierFeatures(output_dim=4, **kwargs)
        self.local_representation2 = Representation_FourierFeatures(output_dim=4, **kwargs)

    def get_params(self):
        params, n_params_repr = self.local_representation1.get_params()
        params2, n_params_repr2 = self.local_representation2.get_params()
        params['params_repr'] += params2['params_repr']
        params['names_params_repr'] += params2['names_params_repr']
        n_params_repr += n_params_repr2

        params_physics = []
        names_params_physics = []

        for name, param in self.named_parameters():
            if "local_representation" not in name:
                params_physics += [param]
                names_params_physics += [name]

        params["params_physics"] = params_physics
        params["names_params_position"] = names_params_physics

        return params, n_params_repr + len(params_physics)

    def update_trafo(self, tspan):
        # Solve ODE for timespan
        if self.use_adjoint:
            y = odeint_adjoint(self.ode, self.x0, tspan)
        else:
            y = odeint(self.ode, self.x0, tspan)

        p1 = y[:, :2]
        p2 = y[:, 2:4]

        self.trafo_from_local1 = lambda x: x.unsqueeze(1) + p1
        self.trafo_to_local1 = lambda x: x.unsqueeze(1) - p1
        self.trafo_from_local2 = lambda x: x.unsqueeze(1) + p2
        self.trafo_to_local2 = lambda x: x.unsqueeze(1) - p2

    def forward(self, x):

        # Transform input points to local coordinate systems
        x_local1 = self.trafo_to_local1(x)
        x_local2 = self.trafo_to_local2(x)

        # Evaluate local mlp for points inside
        output1 = self.local_representation1(x_local1)
        output2 = self.local_representation2(x_local2)

        colors1 = output1[..., :3]
        mask1 = output1[..., 3]
        colors2 = output2[..., :3]
        mask2 = output2[..., 3]

        # Blending
        colors = torch.clip(colors1 * mask1[..., None] + colors2 * mask2[..., None], 0.0, 1.0)

        # Maximum Mask
        mask = torch.max(mask1, mask2)

        masks = {
            'mask_combined': mask,
            'mask1': mask1,
            'mask2': mask2
        }

        return colors, masks

    def render_local_object(self, W, H):
        # Get device of model
        device = next(self.parameters()).device

        # Create coordinates, normalized to the widht
        H = 2*int(H/2)
        W = 2*int(W/2)
        xx = (torch.arange(W, dtype=torch.float32) - W/2) / (W/2)
        yy = (torch.arange(H, dtype=torch.float32)-H/2) / (H/2)
        x_grid, y_grid = torch.meshgrid(xx, yy)
        pixel_coords = torch.stack([x_grid.transpose(0, 1), y_grid.transpose(0, 1)], dim=2).view(-1, 2).to(device)

        output1 = self.local_representation1(pixel_coords)
        output2 = self.local_representation2(pixel_coords)

        colors1 = output1[..., :3].view(H, W, -1)
        mask1 = output1[..., 3].view(H, W)
        colors2 = output2[..., :3].view(H, W, -1)
        mask2 = output2[..., 3].view(H, W)

        return {
            'mask1': mask1,
            'mask2': mask2,
            'colors1': colors1,
            'colors2': colors2
        }


class DynamicBackground(nn.Module):
    def __init__(
        self,
        n_features: int = 256,
        gauss_scale: int = 30,
        n_harmonic_functions_time: int = 4,
        max_frequency_time: int = 20,
        n_layers: int = 8,
        hidden_dim: int = 128,
        input_skips: Tuple[int] = (5, ),
        output_dim: int = 3,
        time_normalization=torch.tensor(1.0),
        **kwargs
    ):
        super().__init__()

        self.register_buffer('time_normalization', time_normalization)
        self.encoder = FourierFeatures(
            n_features=n_features,
            gauss_scale=gauss_scale
        )
        embedding_dim = self.encoder.output_dim

        if n_harmonic_functions_time > 0:
            self.encoder_time = HarmonicEmbedding(
                input_dim=1,
                max_frequency=max_frequency_time,
                n_harmonic_functions=n_harmonic_functions_time,
            )
            embedding_dim_time = self.encoder_time.output_dim
        else:
            self.encoder_time = lambda x: x
            embedding_dim_time = 1

        self.mlp = MLPWithInputSkips(
            n_layers=n_layers,
            input_dim=embedding_dim + embedding_dim_time,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            input_skips=input_skips,
        )

    def forward(self, x, tspan):
        # Input:    x [N, 2]
        #           t [n_tsteps]
        # Output: Shape colors [N, n_tsteps, 3]

        # Encode input
        encoded_input = self.encoder(x)
        encoded_time = self.encoder_time(tspan.unsqueeze(1) / self.time_normalization)

        # Combine spatial and time dimension
        # target: input [N * n_tsteps, encoded dim + 1]
        input = torch.cat([
            encoded_input.repeat_interleave(tspan.shape[0], dim=0),
            encoded_time.repeat(x.shape[0], 1)
        ], dim=1)

        # Evaluate the colors
        colors = self.mlp(input).reshape(x.shape[0], len(tspan), -1)

        return colors

    def render_image(self, W, H, tspan, chunk_size=4096, normalize_by_H=False):
        is_training_initially = self.training
        self.eval()

        dim_time = tspan.shape

        # Get device of model
        device = next(self.parameters()).device

        # Create coordinates, normalized to the widht
        if normalize_by_H:
            normalization = H
        else:
            normalization = W
        xx = torch.arange(W, dtype=torch.float32) / normalization
        yy = torch.arange(H, dtype=torch.float32) / normalization
        x_grid, y_grid = torch.meshgrid(xx, yy)
        pixel_coords = torch.stack([x_grid.transpose(0, 1), y_grid.transpose(0, 1)], dim=2).view(-1, 2).to(device)
        n_pixels = pixel_coords.shape[0]

        n_chunks = math.ceil(n_pixels / chunk_size)
        image = torch.zeros(n_pixels, *dim_time, 3, device=device)

        for i in range(n_chunks):
            start = i*chunk_size
            end = min(start+chunk_size, n_pixels)
            cur_inds = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=device
            )

            # Render current pixels
            with torch.no_grad():
                colors = self.forward(pixel_coords[cur_inds, :], tspan)
                image[cur_inds, ...] = colors

        # Set back to training
        if is_training_initially:
            self.train()

        # Reshape image and return
        output = {}
        output["Image"] = image.transpose(0, 1).view(*dim_time, H, W, -1)

        return output


class Representation_FourierFeatures(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.ff = FourierFeatures(**kwargs)
        self.mlp = MLPWithInputSkips(input_dim=self.ff.output_dim, **kwargs)

    def forward(self, x):
        encoded_input = self.ff(x)
        return self.mlp(encoded_input)

    def get_params(self):
        params_mlp = []
        names_params_mlp = []

        for name, param in self.named_parameters():
            params_mlp += [param]
            names_params_mlp += [name]

        return {"params_repr": params_mlp,
                "names_params_repr": names_params_mlp}, len(params_mlp)


class MLPWithInputSkips(nn.Module):
    def __init__(
        self,
        output_dim: int = 3,
        n_layers: int = 8,
        hidden_dim: int = 128,
        input_skips: Tuple[int] = None,
        input_dim: int = 2,
        use_sigmoid_output: bool = True,
        **kwargs
    ):
        super().__init__()

        # Set to -1 to be used in the later logic
        if input_skips is None:
            input_skips = -1

        if isinstance(input_skips, int):
            input_skips = (input_skips,)

        layers = []

        for i_layer in range(n_layers):
            activation = torch.nn.ReLU(True)
            if i_layer == 0:
                cur_input_dim = input_dim
                cur_output_dim = hidden_dim
            elif i_layer in input_skips:
                cur_input_dim = hidden_dim + input_dim
                cur_output_dim = hidden_dim
            elif i_layer == n_layers-1:
                cur_input_dim = hidden_dim
                cur_output_dim = output_dim
                if use_sigmoid_output:
                    activation = torch.nn.Sigmoid()
            else:
                cur_input_dim = hidden_dim
                cur_output_dim = hidden_dim

            cur_layer = nn.Linear(cur_input_dim, cur_output_dim)
            nn.init.xavier_uniform_(cur_layer.weight.data)
            layers.append(torch.nn.Sequential(cur_layer, activation))

        self.layers = nn.ModuleList(layers)
        self.input_skips = input_skips

    def forward(self, input):
        output = input

        for i_layer, layer in enumerate(self.layers):
            if i_layer in self.input_skips:
                output = torch.cat([output, input], dim=-1)

            output = layer(output)

        return output


class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        max_frequency: float = 1.0,
        min_frequency: float = 1.0,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
        **kwargs
    ):
        super().__init__()

        if logspace:
            frequency_scalings = torch.exp(torch.linspace(
                torch.log(torch.tensor(min_frequency, dtype=torch.float32)),
                torch.log(torch.tensor(max_frequency, dtype=torch.float32)),
                n_harmonic_functions
            ))
        else:
            frequency_scalings = torch.linspace(
                min_frequency, max_frequency, n_harmonic_functions
            )

        self.register_buffer('_frequencies', 2 * math.pi * frequency_scalings)

        if logspace:
            frequency_scalings = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32
            )
        else:
            frequency_scalings = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32
            )

        self.register_buffer('_frequencies', omega_0 * frequency_scalings)
        self.inclue_input = include_input
        self.output_dim = n_harmonic_functions * 2 * input_dim + input_dim

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.inclue_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim=2,
        n_features=256,
        gauss_scale=10.0,
        **kwargs
    ):
        super().__init__()
        self.register_buffer('B', gauss_scale * torch.randn(n_features, input_dim))
        self.output_dim = 2*n_features

    def forward(self, x: torch.Tensor):
        # Shape x: (N, 2)
        # Shape B: (N_features, 2)
        x_proj = (2.*math.pi * x) @ self.B.transpose(0, 1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
