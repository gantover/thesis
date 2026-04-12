import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from ADM.models.diffusion import Model
from ADM.models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from ADM.models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from ADM.utils import (
    compute_alpha,
    inverse_data_transform,
    singlestep_ddim_sample,
    parse_args_and_config,
    seed_everything,
    get_beta_schedule,
    preprocess_la_adm,
    postprocess_la_adm,
)
from diffusion_laplace import LaplaceDataset, DiffusionLLDiagLaplace


def main(args, config):

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # set random seed
    seed_everything(args.seed)
    fixed_xT = torch.randn(
        [
            args.total_n_sample,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        ]
    )
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError(
            "Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(
                total_n_samples, args.sample_batch_size
            )
        )
    n_rounds = total_n_samples // args.sample_batch_size
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(args.sample_batch_size, n_rounds))
    else:
        fixed_classes = torch.randint(
            low=args.fixed_class,
            high=args.fixed_class + 1,
            size=(args.sample_batch_size, n_rounds),
        ).to(device)

    ######  initialize diffusion and model(unet) ##########
    if config.model.model_type == "guided_diffusion":
        model = GuidedDiffusion_Model(
            image_size=config.model.image_size,
            in_channels=config.model.in_channels,
            model_channels=config.model.model_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attention_resolutions,
            dropout=config.model.dropout,
            channel_mult=config.model.channel_mult,
            conv_resample=config.model.conv_resample,
            dims=config.model.dims,
            num_classes=config.model.num_classes,
            use_checkpoint=config.model.use_checkpoint,
            use_fp16=config.model.use_fp16,
            num_heads=config.model.num_heads,
            num_head_channels=config.model.num_head_channels,
            num_heads_upsample=config.model.num_heads_upsample,
            use_scale_shift_norm=config.model.use_scale_shift_norm,
            resblock_updown=config.model.resblock_updown,
            use_new_attention_order=config.model.use_new_attention_order,
        )

    else:
        model = Model(config)

    model = model.to(device)
    map_location = {"cuda:%d" % 0: "cuda:%d" % args.device}

    if "ckpt_dir" in config.model.__dict__.keys():
        ckpt_dir = os.path.expanduser(config.model.ckpt_dir)
        states = torch.load(ckpt_dir, map_location=map_location)
        if config.model.model_type == "improved_ddpm" or config.model.model_type == "guided_diffusion":
            model.load_state_dict(states, strict=True)
            if config.model.use_fp16:
                model.convert_to_fp16()
        else:
            modified_states = {}
            for key, value in states[0].items():
                modified_key = key[7:]
                modified_states[modified_key] = value
            model.load_state_dict(modified_states, strict=True)

    classifier = None
    classifier_grad_batch_size = (
        args.classifier_grad_batch_size
        if args.classifier_grad_batch_size is not None
        else args.sample_batch_size
    )
    if classifier_grad_batch_size <= 0:
        raise ValueError("classifier_grad_batch_size must be a positive integer.")

    classifier_scale = (
        args.classifier_scale
        if args.classifier_scale is not None
        else getattr(config.sampling, "classifier_scale", 1.0)
    )
    if args.guidance_mode == "classifier":
        if config.model.model_type != "guided_diffusion":
            raise ValueError("Classifier guidance is only supported for guided_diffusion model_type in this script.")
        if not hasattr(config, "classifier"):
            raise ValueError("Missing classifier section in config for classifier guidance mode.")

        classifier = GuidedDiffusion_Classifier(
            image_size=config.classifier.image_size,
            in_channels=config.classifier.in_channels,
            model_channels=config.classifier.model_channels,
            out_channels=config.classifier.out_channels,
            num_res_blocks=config.classifier.num_res_blocks,
            attention_resolutions=config.classifier.attention_resolutions,
            channel_mult=config.classifier.channel_mult,
            use_fp16=config.classifier.use_fp16,
            num_head_channels=config.classifier.num_head_channels,
            use_scale_shift_norm=config.classifier.use_scale_shift_norm,
            resblock_updown=config.classifier.resblock_updown,
            pool=config.classifier.pool,
        ).to(device)

        classifier_ckpt_dir = os.path.expanduser(config.classifier.ckpt_dir)
        classifier_states = torch.load(classifier_ckpt_dir, map_location=map_location)
        if isinstance(classifier_states, dict) and "state_dict" in classifier_states:
            classifier_states = classifier_states["state_dict"]
        try:
            classifier.load_state_dict(classifier_states, strict=True)
        except RuntimeError:
            # Support checkpoints saved from DDP wrappers.
            stripped_state = {
                (k[7:] if k.startswith("module.") else k): v for k, v in classifier_states.items()
            }
            classifier.load_state_dict(stripped_state, strict=True)
        if config.classifier.use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

    la_dataset = LaplaceDataset(
        device,
        config.data.path,
        image_size=config.model.image_size,
        train_la_data_size=args.train_la_data_size,
    )
    la_dataloader = torch.utils.data.DataLoader(la_dataset, batch_size=args.train_la_batch_size, shuffle=True)

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]

    # # print model layers
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel()}")

    _preprocess_la_adm = lambda x, y, device: preprocess_la_adm(x, y, betas, betas.shape[0], device)
    la = DiffusionLLDiagLaplace(
        model,
        f_preprocess_la_input=_preprocess_la_adm,
        f_postprocess_la_output=postprocess_la_adm,
        last_layer_name="out.2",
        temperature=config.laplace.temperature,
    )
    print(f"la.temperature: {la.temperature}")
    la.fit(la_dataloader)
    last_layers = la.sample(args.mc_size) # S will be the number of MC samples, D will be the number of parameters in the last layer

    # add MAP model
    last_layers = torch.concat([la.mean[None, :], last_layers], dim=0)

    print(last_layers.shape)
    print(type(la))

    if args.skip_type == "uniform":
        skip = num_timesteps // args.timesteps
        seq = range(0, num_timesteps, skip)
    elif args.skip_type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), args.timesteps) ** 2
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError

    exp_dir = f"{args.exp_path}/{config.data.dataset}/ddim_{args.guidance_mode}_fixed_class{args.fixed_class}_train%{args.train_la_data_size}_step{args.timesteps}_S{args.mc_size}_epi_unc_{args.seed}/"
    os.makedirs(exp_dir, exist_ok=True)
    np.save(os.path.join(exp_dir, "classes.npy"), fixed_classes.cpu().numpy())

    S, D = last_layers.shape
    for s in range(S):

        os.makedirs(exp_dir + f"{s}/imgs", exist_ok=True)
        img_count = 0

        with torch.no_grad():

            # overwrite the parameters of the last layer with the sampled layer
            model_params = parameters_to_vector(model.parameters())
            model_params[-D:] = last_layers[s]
            vector_to_parameters(model_params, model.parameters())

            for loop in tqdm.tqdm(range(n_rounds), desc="Generating image samples for FID evaluation."):

                if config.sampling.cond_class:
                    classes = fixed_classes[:, loop].to(device)
                else:
                    classes = None

                if classes is None:
                    model_kwargs = {}
                else:
                    model_kwargs = {"y": classes}

                def predict_eps(x_t, t_discrete):
                    eps_t = model.forward_no_cfg(x_t, t_discrete, **model_kwargs)
                    if classifier is None:
                        return eps_t

                    if classes is None:
                        raise ValueError("Classifier guidance requires class labels.")

                    cond_grad_chunks = []
                    n_batch = x_t.shape[0]
                    for start in range(0, n_batch, classifier_grad_batch_size):
                        end = min(start + classifier_grad_batch_size, n_batch)
                        with torch.enable_grad():
                            x_in = x_t[start:end].detach().requires_grad_(True)
                            t_in = t_discrete[start:end]
                            y_in = classes[start:end]
                            logits = classifier(x_in, t_in)
                            log_probs = F.log_softmax(logits, dim=-1)
                            selected = log_probs[torch.arange(logits.shape[0], device=x_t.device), y_in.view(-1)]
                            cond_grad_chunk = torch.autograd.grad(selected.sum(), x_in)[0]
                        cond_grad_chunks.append(cond_grad_chunk)

                    cond_grad = torch.cat(cond_grad_chunks, dim=0)

                    at = compute_alpha(betas, t_discrete.long())
                    sigma_t = (1 - at).sqrt()
                    return eps_t - sigma_t * (classifier_scale * cond_grad)

                xT = fixed_xT[loop * args.sample_batch_size : (loop + 1) * args.sample_batch_size, :, :, :].to(device)
                xt_next = xT
                t_discrete = (torch.ones(args.sample_batch_size) * seq[args.timesteps - 1]).to(xT.device).to(torch.int64)
                eps_mu_t = predict_eps(xT, t_discrete)

                for timestep in range(args.timesteps - 1, 0, -1):
                    xt_next = singlestep_ddim_sample(betas, xt_next, seq, timestep, eps_mu_t)
                    t_discrete = (torch.ones(args.sample_batch_size) * seq[timestep - 1]).to(xt_next.device).to(torch.int64)
                    eps_mu_t = predict_eps(xt_next, t_discrete)

                x = inverse_data_transform(config, xt_next)

                # torch.save(x.cpu().numpy(), os.path.join(exp_dir + f"{s}/", f"{loop}.pt"))

                x = x.cpu().numpy()
                for i in range(x.shape[0]):
                    img = x[i].transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_pil.save(os.path.join(exp_dir + f"{s}/imgs", f"{img_count:05d}.png"))
                    img_count += 1

    return exp_dir


if __name__ == "__main__":
    args, config = parse_args_and_config()
    exp_dir = main(args, config)
    print(exp_dir, end="")
