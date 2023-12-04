import os
import logging
import time
import glob
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
from models.diffusion import Model, Model_gradient_checkpointing
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, KID, KID_rbf
from functions.clip_features import CLIP_fx
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from functions.denoising import generalized_steps, generalized_steps_diff, generalized_steps_gp
from piq.feature_extractors import InceptionV3
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from fid_utils.precalc_fid import calculate_activation_statistics_for_dataloader

EPS = 1e-20

import torchvision.utils as tvu
from functions.image_utils import generate_sample_sheet, generate_sample_sheet_4, generate_sample_sheet_8
 



def plot_grad_flow(named_parameters, i):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #print(f'{n}{p.grad.abs().mean().cpu().detach().numpy()}')
            ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
            max_grads.append(p.grad.abs().max().cpu().detach().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout() 
    plt.savefig(f"./gradients/gradient_{i}.png", pad_inches = 9.0)

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
                
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            print(f"seq {seq}")
           
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


    def sample_interpolation(self):
        config = self.config

        model = Model(self.config)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        states = torch.load("./ckpt_400.pth")
        model.load_state_dict(states[0])
        
        model.eval()

        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        print(f"using timesteps {timesteps}")


        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                samples = generalized_steps(x[i:i+8], timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                xs.append(samples)
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], f"./sample_ddim_{i}.png")
        
    def generate_samples(self):

        args, config = self.args, self.config
        
        model = Model(self.config)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        states = torch.load("./logs/church20_CLIP/ckpt_700.pth")
        model.load_state_dict(states[0])
        
        model.eval()

        img_id = 0
        print(f"starting from image {img_id}")
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        print(f"using timesteps {timesteps}")
        with torch.no_grad():
            for k in tqdm.tqdm(
                range(15), desc="Generating image samples for FID evaluation."
            ):
                n =25
                e = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                samples_list = []

                #Sampling from the ddim model 
                samples = generalized_steps(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                samples = inverse_data_transform(self.config, samples)
     
                for i in range(25):
                    samples_list.append(samples[i].permute(1,2,0).cpu().detach().numpy())
                generate_sample_sheet_8(samples_list, k, config.data.image_size)
                

    

    def test_FID(self):


        args, config = self.args, self.config
        
        model = Model(self.config)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        states = torch.load("./church10/ckpt_450.pth")
        model.load_state_dict(states[0])
        
        model.eval()

        img_id = 0
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        # seq = (
        #             np.linspace(
        #                 0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
        #             )
        #             ** 2
        #         )
        # timesteps = [int(s) for s in list(seq)]
        all_samples = []
        print(f"using timesteps {timesteps}")
        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                e = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                
                #Sampling from the ddim model 
                samples = generalized_steps(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                samples = inverse_data_transform(self.config, samples)

                
                for i in range(n):
                    tvu.save_image(
                        samples[i], "./church_test/{}.png".format(img_id)
                    )
                    img_id += 1
        
            print("Sampling complete")

       
 

    def knn_features(self):


        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        model = Model(self.config)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        states = torch.load("./celeb5_CLIP_cub/ckpt_400.pth")
        model.load_state_dict(states[0])
        
        model.eval()

        # Load the feature extractor
        feature_extractor = CLIP_fx("ViT-B/32",self.device).to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()

        img_id = 0
        print(f"starting from image {img_id}")
        n_rounds = 1
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        
        # cifar_10_data = []
        # for i, (x, y) in enumerate(train_loader):
        #     x = x.to(self.device)
        #     x = feature_extractor(x)
        #     cifar_10_data.append(x)
        # cifar_10_data = torch.cat(cifar_10_data, dim=0).to(self.device)
        # print(cifar_10_data.shape)

        print(f"using timesteps {timesteps}")
        with torch.no_grad():
            for _ in range(n_rounds):
                n = 20
                e = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                #Sampling from the ddim model 
                samples = generalized_steps(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                samples = inverse_data_transform(self.config, samples)
                samples_feat = feature_extractor(samples)

                distances_all = []
                images_all = []
  
                # Iterate over all images in the dataset
                for i in tqdm.tqdm(range(n)):
                    # Calculate the distance between the generated sample and the dataset images in CLIP feature space
                    distances = []
                    images = []
                    for k, (x, y) in tqdm.tqdm(enumerate(train_loader)):
                        x = x.to(self.device)
                        feat_x = feature_extractor(x)
                        dist = torch.nn.functional.pairwise_distance(samples_feat[i], feat_x)
                        distances.append(dist)
                        images.append(x)
                        
                    
                    distances = torch.cat(distances, dim=0)
                    images = torch.cat(images, dim=0)
                    
                    dist, index = torch.topk(distances, k=5, largest=False, sorted = True)
                    

                    top_images = torch.index_select(images, 0, index)
                    images_all.append(top_images)
                images_all = torch.cat(images_all, dim=0)

                # Save the generated image and the closest images from the dataset
                for i in range(n):
                    tvu.save_image(
                        samples[i], "./DDIM/NN_vis/ddim_celeb/{}.png".format(img_id)
                    )
                    for j in range(5):
                        tvu.save_image(
                            images_all[i*5+j], "./DDIM/NN_vis/ddim_celeb/{}_{}.png".format(img_id,j)
                        )
                    img_id += 1

                


                

                

           
    def compute_stats_lsun(self):

        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        #dataset_all = torch.utils.data.ConcatDataset([dataset, test_dataset])
        train_loader = data.DataLoader(
            dataset,
            batch_size=500,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device)
        model = torch.nn.DataParallel(model)

        print("Calculate FID stats..", end=" ", flush=True)
        with torch.no_grad():
            mu, sigma = calculate_activation_statistics_for_dataloader(model, train_loader, cuda=True, verbose=True)

        np.savez_compressed("./DDIM/fid_utils/lsun_fid_statistics.npz", mu=mu, sigma=sigma)
        print("Finished")

    def train_cifar(self):
        
        degree = 3
        use_checkpointing = False
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        n_accumulation = 1
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError

        if use_checkpointing:
            model = Model_gradient_checkpointing(self.config)
        else:
            model = Model(self.config)
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()

        # Load the feature extractor
        feature_extractor = InceptionV3().to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        
        loss_kid = KID(degree = 3)
        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0
        rand = torch.randn(16, 3, config.data.image_size, config.data.image_size).to(self.device)
        # cifar_10_data = []
        # for i, (x, y) in enumerate(train_loader):
        #         cifar_10_data.append(x)
        # cifar_10_data = torch.cat(cifar_10_data, dim=0).to(self.device)
        # print(cifar_10_data.shape)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        # seq = (
        #             np.linspace(
        #                 0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
        #             )
        #             ** 2
        #         )
        # timesteps = [int(s) for s in list(seq)]
        print(f"Using timesteps: {timesteps}")
        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                #Sampling from the ddim model 
                samples = generalized_steps_diff(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)[0].view(n, -1).type(torch.float64)
                x_feat = feature_extractor(x)[0].view(n, -1).type(torch.float64)

                loss = loss_kid(samples_feat, x_feat)/n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                
                loss.backward()
                if step % self.config.training.validation_freq == 0 or step == 1:
                    #plot_grad_flow(model.named_parameters(), step)
                    with torch.no_grad():
                        samples_list = []
                        sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                        sequence = inverse_data_transform(self.config, sequence)
                        for i in range(16):
                            samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                        generate_sample_sheet(samples_list, step, config.data.image_size)
                      
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                
                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
  
    def generate_reference(self):

        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        dataset_all = torch.utils.data.ConcatDataset([dataset, test_dataset])
        train_loader = data.DataLoader(
            dataset,
            batch_size=9,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        for k, (x, y) in enumerate(train_loader):
            samples = x.to(self.device)
            samples_list = []
            for i in range(9):
                samples_list.append(samples[i].permute(1,2,0).cpu().detach().numpy())
            generate_sample_sheet_8(samples_list, k, config.data.image_size)
            if(k==30):
                break

        return 

    def train_celeba(self):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        n_accumulation = 1
        model = Model(self.config)
       
        model.load_state_dict(torch.load('./DDIM/ckpt_celebA.pth', map_location=self.device)[4])
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()

        # Load the feature extractor
        feature_extractor = InceptionV3().to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        
        loss_kid = KID(degree = 3)
        optimizer = get_optimizer(self.config, model.parameters())
        start_epoch, step = 0, 0
        rand = torch.randn(16, 3, config.data.image_size, config.data.image_size).to(self.device)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        print(f"Using timesteps: {timesteps}")
        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                
                #Sampling from the ddim model 
                samples = generalized_steps_gp(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]

                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)[0].view(n, -1)
                x_feat = feature_extractor(x)[0].view(n, -1)
                loss = loss_kid(samples_feat, x_feat)/ n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                loss.backward()
                if step % self.config.training.validation_freq == 0 or step == 1:
                    #plot_grad_flow(model.named_parameters(), i)
                    with torch.no_grad():
                        samples_list = []
                        sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                        sequence = inverse_data_transform(self.config, sequence)
                        for i in range(16):
                            samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                        generate_sample_sheet(samples_list, step, config.data.image_size)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()


    def train_lsun(self):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        n_accumulation = 4

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError


        model = Model(self.config)
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()

        # Load the feature extractor
        feature_extractor = InceptionV3().to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        
        loss_kid = KID(degree=3)
        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0
        rand = torch.randn(4, 3, config.data.image_size, config.data.image_size).to(self.device)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)

        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                #Sampling from the ddim model 
                samples = generalized_steps_gp(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)[0].view(n, -1)
                x_feat = feature_extractor(x)[0].view(n, -1)

                loss = loss_kid(samples_feat, x_feat) / n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                loss.backward()
                if step % self.config.training.validation_freq == 0 or step == 1:
                    with torch.no_grad():
                        samples_list = []
                        sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                        sequence = inverse_data_transform(self.config, sequence)
                        for i in range(4):
                            samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                        generate_sample_sheet_4(samples_list, step, config.data.image_size)
                      
                #if you want to clip the gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
  

    def train_cifar_CLIP(self):
        
        use_checkpointing = False
        kernel = None
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        n_accumulation = 2
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError

        if use_checkpointing:
            model = Model_gradient_checkpointing(self.config)
        else:
            model = Model(self.config)

        
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()

        # Load the feature extractor
        feature_extractor = CLIP_fx("ViT-B/32",self.device).to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        if kernel == "rbf":
            loss_kid = KID_rbf()
        else:
            loss_kid = KID(degree = 3)
        optimizer = get_optimizer(self.config, model.parameters())
        optimizer.zero_grad()
        start_epoch, step = 0, 0
        rand = torch.randn(16, 3, config.data.image_size, config.data.image_size).to(self.device)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        print(f"Using timesteps: {timesteps}")
        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                #Sampling from the ddim model 
                samples = generalized_steps_diff(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)#[0].view(n, -1)
                x_feat = feature_extractor(x)#[0].view(n, -1)

                samples_feat = samples_feat.type(torch.float32)
                x_feat = x_feat.type(torch.float32)

                #print(samples_feat.mean(), x_feat.mean())
                
               # print(x_feat.shape)
               # print(samples_feat.shape)

                loss = loss_kid(samples_feat, x_feat)/n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                loss.backward()
                if step % self.config.training.validation_freq == 0 or step == 1:
                    plot_grad_flow(model.named_parameters(), step)
                    with torch.no_grad():
                        samples_list = []
                        sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                        sequence = inverse_data_transform(self.config, sequence)
                        for i in range(16):
                            samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                        generate_sample_sheet(samples_list, step, config.data.image_size)
                      
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                
                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
  

    def train_celeba_CLIP(self):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        n_accumulation = 1
        model = Model(self.config)
       
        model.load_state_dict(torch.load('./DDIM/ckpt_celebA.pth', map_location=self.device)[4])
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()


        # Load the feature extractor
        feature_extractor = CLIP_fx("ViT-B/32",self.device).to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        
        
        loss_kid = KID(degree = 3)
        optimizer = get_optimizer(self.config, model.parameters())
        optimizer.zero_grad()
        start_epoch, step = 0, 0
        rand = torch.randn(16, 3, config.data.image_size, config.data.image_size).to(self.device)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)
        print(f"Using timesteps: {timesteps}")


        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                
                #Sampling from the ddim model 
                samples = generalized_steps_gp(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]

                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)
                x_feat = feature_extractor(x)

                samples_feat = samples_feat.type(torch.float32)
                x_feat = x_feat.type(torch.float32)

                loss = loss_kid(samples_feat, x_feat)/ n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                loss.backward()
                if step % self.config.training.validation_freq == 0 or step == 1:
                    #plot_grad_flow(model.named_parameters(), i)
                    with torch.no_grad():
                        samples_list = []
                        sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                        sequence = inverse_data_transform(self.config, sequence)
                        for i in range(16):
                            samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                        generate_sample_sheet(samples_list, step, config.data.image_size)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()


    def train_lsun_CLIP(self):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        n_accumulation = 4

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError


        model = Model(self.config)
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.train()

        # Load the feature extractor
        feature_extractor = CLIP_fx("ViT-B/32",self.device).to(self.device)
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()
        
        loss_kid = KID(degree=3)
        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0
        #rand = torch.randn(4, 3, config.data.image_size, config.data.image_size).to(self.device)
        skip = self.num_timesteps // self.args.timesteps
        timesteps = range(0, self.num_timesteps, skip)

        for epoch in range(start_epoch, config.training.n_epochs):
            
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                #Sampling from the ddim model 
                samples = generalized_steps_gp(e, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                x=inverse_data_transform(self.config, x)
                samples=inverse_data_transform(self.config, samples)

                #Loss computation KID
                samples_feat = feature_extractor(samples)
                x_feat = feature_extractor(x)

                samples_feat = samples_feat.type(torch.float32)
                x_feat = x_feat.type(torch.float32)
                loss = loss_kid(samples_feat, x_feat) / n_accumulation

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                loss.backward()
                #if step % self.config.training.validation_freq == 0 or step == 1:
                    # with torch.no_grad():
                    #     samples_list = []
                    #     sequence = generalized_steps_diff(rand, timesteps, model, self.betas, eta=self.args.eta)[0][-1]
                    #     sequence = inverse_data_transform(self.config, sequence)
                    #     for i in range(4):
                    #         samples_list.append(sequence[i].permute(1,2,0).cpu().detach().numpy())
                    #     generate_sample_sheet_4(samples_list, step, config.data.image_size)
                      
                #if you want to clip the gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if((i+1) % n_accumulation == 0 or i == len(train_loader)-1):
                    optimizer.step()
                    optimizer.zero_grad()

                if step % self.config.training.validation_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
