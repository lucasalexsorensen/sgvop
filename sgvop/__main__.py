import os
import pickle
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from .config import get_config_parser
from .loading import load_blender_data, load_llff_data
from .model import create_nerf, sigma_sparsity_loss, total_variation_loss
from .utils import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    parser = get_config_parser()
    args = parser.parse_args()

    # load dataset
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=0.75, spherify=args.spherify
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        args.bounding_box = bounding_box
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)]
        )

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0

        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        args.bounding_box = bounding_box
        print("BOUNDING_BOX", args.bounding_box)
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    # camera intrinsics
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    # create log dir and copy the config file
    basedir, expname = Utils.make_log_dir(args)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    # obtained near/far bounds from scene data
    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # move poses to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    poses = torch.Tensor(poses).to(device)

    N_iters = 50000 + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        if args.N_rand is not None:
            rays_o, rays_d = Utils.get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    ),
                    -1,
                )
                if i == start:
                    print(
                        f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                    )
            else:
                coords = torch.stack(
                    torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                    -1,
                )  # (H, W, 2)

            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(
                coords.shape[0], size=[args.N_rand], replace=False
            )  # (args.N_rand,)
            select_coords = coords[select_inds].long()  # (args.N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (args.N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (args.N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (args.N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = Utils.render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        optimizer.zero_grad()
        img_loss = Utils.img2mse(rgb, target_s)
        trans = extras["raw"][..., -1]
        loss = img_loss
        psnr = Utils.mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = Utils.img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = Utils.mse2psnr(img_loss0)

        sparsity_loss = args.sparse_loss_weight * (
            extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum()
        )
        loss = loss + sparsity_loss

        # add Total Variation loss
        n_levels = render_kwargs_train["embed_fn"].L
        min_res = render_kwargs_train["embed_fn"].N_min
        max_res = render_kwargs_train["embed_fn"].N_max
        log2_hashmap_size = render_kwargs_train["embed_fn"].log2_T
        TV_loss = sum(
            total_variation_loss(
                render_kwargs_train["embed_fn"].embeddings[i],
                min_res,
                max_res,
                i,
                log2_hashmap_size,
                n_levels=n_levels,
            )
            for i in range(n_levels)
        )
        loss = loss + args.tv_loss_weight * TV_loss
        if i > 1000:
            args.tv_loss_weight = 0.0

        loss.backward()
        # pdb.set_trace()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        t = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                    "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
                    "embed_fn_state_dict": render_kwargs_train["embed_fn"].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = Utils.render_path(
                    render_poses, hwf, K, args.chunk, render_kwargs_test
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, "{}_spiral_{:06d}_".format(expname, i))
            imageio.mimwrite(moviebase + "rgb.mp4", Utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", Utils.to8b(disps / np.max(disps)), fps=30, quality=8
            )

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                Utils.render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    K,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {"losses": loss_list, "psnr": psnr_list, "time": time_list}
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.FloatTensor")
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    train()
