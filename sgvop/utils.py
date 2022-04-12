import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm


class Utils:
    @staticmethod
    def img2mse(x, y, mask=None):
        if mask is None:
            return torch.mean((x - y) ** 2)
        return torch.mean((mask * (x - y)) ** 2)

    mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    @staticmethod
    def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0.0, 1.0, N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    @staticmethod
    def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        # sigma_loss = sigma_sparsity_loss(raw[...,3])
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
            )[:, :-1]
        )
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        # Calculate weights sparsity loss
        mask = weights.sum(-1) > 0.5
        entropy = Categorical(probs=weights + 1e-5).entropy()
        sparsity_loss = entropy * mask

        return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss

    @staticmethod
    def ndc_rays(H, W, focal, near, rays_o, rays_d):
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1.0 + 2.0 * near / rays_o[..., 2]

        d0 = (
            -1.0
            / (W / (2.0 * focal))
            * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        )
        d1 = (
            -1.0
            / (H / (2.0 * focal))
            * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        )
        d2 = -2.0 * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    @staticmethod
    def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = Utils.render_rays(rays_flat[i : i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    @staticmethod
    def get_rays(H, W, K, c2w):
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
        )  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack(
            [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
        )
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(
            dirs[..., np.newaxis, :] * c2w[:3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    @staticmethod
    def render(
        H,
        W,
        K,
        chunk=1024 * 32,
        rays=None,
        c2w=None,
        ndc=True,
        near=0.0,
        far=1.0,
        use_viewdirs=False,
        c2w_staticcam=None,
        **kwargs,
    ):
        """Render rays
        Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        focal: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
        """
        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = Utils.get_rays(H, W, K, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = Utils.get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = Utils.ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = Utils.batchify_rays(rays, chunk, **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ["rgb_map", "disp_map", "acc_map"]
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    @staticmethod
    def render_rays(
        ray_batch,
        network_fn,
        network_query_fn,
        N_samples,
        embed_fn=None,
        retraw=False,
        lindisp=False,
        perturb=0.0,
        N_importance=0,
        network_fine=None,
        white_bkgd=False,
        raw_noise_std=0.0,
        verbose=False,
        pytest=False,
    ):
        """Volumetric rendering.
        Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
            in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
        if not lindisp:
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.0:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = Utils.raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

        if N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0, sparsity_loss_0 = (
                rgb_map,
                disp_map,
                acc_map,
                sparsity_loss,
            )

            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = Utils.sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = (
                rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            )  # [N_rays, N_samples + N_importance, 3]

            run_fn = network_fn if network_fine is None else network_fine
            #         raw = run_network(pts, fn=run_fn)
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = Utils.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
            )

        ret = {
            "rgb_map": rgb_map,
            "disp_map": disp_map,
            "acc_map": acc_map,
            "sparsity_loss": sparsity_loss,
        }
        if retraw:
            ret["raw"] = raw
        if N_importance > 0:
            ret["rgb0"] = rgb_map_0
            ret["disp0"] = disp_map_0
            ret["acc0"] = acc_map_0
            ret["sparsity_loss0"] = sparsity_loss_0
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        # for k in ret:
        #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
        #         print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    @staticmethod
    def render_path(
        render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0
    ):

        H, W, focal = hwf

        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor
            focal = focal / render_factor

        rgbs = []
        disps = []
        psnrs = []

        t = time.time()
        for i, c2w in enumerate(tqdm(render_poses)):
            print(i, time.time() - t)
            t = time.time()
            rgb, disp, acc, _ = Utils.render(
                H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
            )
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())
            if i == 0:
                print(rgb.shape, disp.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10.0 * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
                psnrs.append(p)

            if savedir is not None:
                rgb8 = Utils.to8b(rgbs[-1])
                filename = os.path.join(savedir, "{:03d}.png".format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)
        if gt_imgs is not None and render_factor == 0:
            print("Avg PSNR over Test set: ", sum(psnrs) / len(psnrs))

        return rgbs, disps

    @staticmethod
    def make_log_dir(args):
        basedir = args.basedir
        args.expname += "_hashXYZ"
        args.expname += "_sphereVIEW"
        args.expname += "_fine" + str(args.finest_res) + "_log2T" + str(args.log2_hashmap_size)
        args.expname += "_lr" + str(args.lrate) + "_decay" + str(args.lrate_decay)
        args.expname += "_RAdam"
        if args.sparse_loss_weight > 0:
            args.expname += "_sparse" + str(args.sparse_loss_weight)
        args.expname += "_TV" + str(args.tv_loss_weight)
        # args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
        expname = args.expname

        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, "args.txt")
        with open(f, "w") as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write("{} = {}\n".format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, "config.txt")
            with open(f, "w") as file:
                file.write(open(args.config, "r").read())

        return basedir, expname
