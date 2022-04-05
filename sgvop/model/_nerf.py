import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding import MultiresolutionHashEncoder, SphericalHarmonicsEncoder
from ._optim import RAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def _run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = _batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


# Smaller NeRF (since hash/SH embeddings are always used)
class NeRFSmall(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=4,
        hidden_dim_color=64,
        input_ch=3,
        input_ch_views=3,
    ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


def _get_embedder(args, type: str):
    if type == "identity":
        return nn.Identity(), 3
    elif type == "hash":
        embed = MultiresolutionHashEncoder(
            bounding_box=args.bounding_box,
            log2_T=args.log2_hashmap_size,
            N_max=args.finest_res,
        )
        out_dim = embed.out_dim
    elif type == "sh":
        embed = SphericalHarmonicsEncoder()
        out_dim = embed.out_dim
    return embed, out_dim


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = _get_embedder(args, "hash")
    # hashed embedding table
    embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = _get_embedder(args, "sh")

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = NeRFSmall(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRFSmall(
            num_layers=2,
            hidden_dim=64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color=64,
            input_ch=input_ch,
            input_ch_views=input_ch_views,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: _run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    # Create optimizer
    optimizer = RAdam(
        [
            {"params": grad_vars, "weight_decay": 1e-6},
            {"params": embedding_params, "eps": 1e-15},
        ],
        lr=args.lrate,
        betas=(0.9, 0.99),
    )

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])
        embed_fn.load_state_dict(ckpt["embed_fn_state_dict"])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "embed_fn": embed_fn,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
