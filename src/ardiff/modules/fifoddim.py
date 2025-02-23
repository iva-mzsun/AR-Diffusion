"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from einops import repeat

from src.ardiff.modules.utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.betas.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def generate_timestep_matrix(self, total_num_frames, base_num_frames, ar_step, step_template):
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template)
        step_template = np.concatenate([np.array([999]), step_template], axis=0)
        step_template = torch.tensor(step_template, dtype=torch.long)
        pre_row = torch.zeros(total_num_frames, dtype=torch.long)

        # 当前矩阵的最后一列还不全为最后一个时间步时，说明还没有生成完毕
        while torch.all(pre_row == num_iterations) == False: 
            # 生成下一列
            new_row = torch.zeros(total_num_frames, dtype=torch.long)
            for i in range(total_num_frames):
                if i == 0 or pre_row[i-1] == num_iterations: # 首帧 / 前一个视频帧已经完全去噪音
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i-1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            # print(new_row, new_row != pre_row)
            update_mask.append(new_row != pre_row) # 仅更新步数改变了的视频帧，False: 不用更新， True: 需要更新
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row
        
        # 基于update_mask，计算feed mask
        terminal_flag = base_num_frames
        for i in range(0, len(update_mask)):
            if terminal_flag < total_num_frames and \
                update_mask[i][terminal_flag] == True: # 下一个视频帧要更新，则指向下下个视频帧
                terminal_flag += 1
            valid_interval.append((terminal_flag - base_num_frames, terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)
        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               base_num_frames,
               shape, # (batch_size, total_num_frames, n_token_per_frame, n_channel_per_token)
               conditioning=None,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               quantize_denoised=False,
               ardiff_step=None, # [0, 50], 0="full_sequence", 50="auto_regressive"
               **kwargs):
        
        assert ardiff_step >= 0 and ardiff_step <= S, f"ardiff_step must be in [0, {S}]"
        device = self.model.betas.device
        total_num_frames = shape[1]
        img = x_T if x_T is not None else torch.randn(shape, device=device)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        timestep_template = np.flip(self.ddim_timesteps)
        timestep_matrix, step_index, update_masks, valid_intervals = self.generate_timestep_matrix(total_num_frames, base_num_frames, 
                                                                               ardiff_step, timestep_template)
        # print(timestep_matrix, step_index)

        # intermediates = {'x_inter': [img], 'pred_x0': [img]}
        if verbose:
            print(f"Running DDIM Sampling with {S} timesteps and {timestep_matrix.shape[0]} iterations")
            iterator = tqdm(range(timestep_matrix.shape[0]), desc='DDIM Sampler', total=timestep_matrix.shape[0])
            print(f"Current timestep matrix: {timestep_matrix}")
            print(f"Step index: {step_index}")
            print(f"Step update mask: {update_mask}")
            print(f"Step valid interval: {valid_interval}")
        else:
            iterator = list(range(timestep_matrix.shape[0]))

        for i, step in enumerate(iterator):
            update_mask = update_masks[i][None, :, None, None].int().to(device)
            
            index = len(self.ddim_timesteps) - step_index[i]
            ts = repeat(timestep_matrix[i], 't -> b t', b=batch_size)
            ts = ts.long().to(device)

            if mask is not None:
                assert x0 is not None
                #img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                try:
                    img_orig = x0
                    img = img_orig * mask + (1. - mask) * img
                    zero_ts = torch.ones_like(ts)
                    tmp_ts = zero_ts * mask[:, :, 0, 0] + (1. - mask[:, :, 0, 0]) * ts
                    ts = tmp_ts.to(ts.dtype)
                except:
                    import ipdb; ipdb.set_trace()


            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            valid_interval_s, valid_interval_e = valid_intervals[i]
            valid_update_mask = update_mask[:, valid_interval_s : valid_interval_e, :, :]
            valid_img = img[:, valid_interval_s : valid_interval_e, :, :]
            valid_index = index[valid_interval_s : valid_interval_e]
            valid_ts = ts[:, valid_interval_s : valid_interval_e]

            outs = self.p_sample_ddim(valid_img, conditioning, valid_ts, index=valid_index, 
                                      update_mask=valid_update_mask,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            valid_img, pred_x0 = outs
            img[:, valid_interval_s : valid_interval_e, :, :] = valid_img

        return img, None

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, update_mask, 
                      repeat_noise=False, use_original_steps=False, 
                      quantize_denoised=False, temperature=1., noise_dropout=0., 
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # NOTE: Do not update values if the noise step is the same
        # print(f"t: {t.shape} update_mask: {update_mask.shape}, x: {x.shape}, model_output: {model_output.shape}")
        model_output = model_output * update_mask + x * (1 - update_mask)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        elif self.model.parameterization == "x0":
            e_t = self.model._predict_eps_from_xstart(x, t, model_output)
        elif self.model.parameterization == "eps":
            e_t = model_output
        else:
            raise NotImplementedError(self.model.parameterization)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = repeat(torch.tensor(alphas[index], dtype=x.dtype), 
                     't -> b t l c', b=b, l=1, c=1).to(device)
        a_prev = repeat(torch.tensor(alphas_prev[index], dtype=x.dtype), 
                        't -> b t l c', b=b, l=1, c=1).to(device)
        sigma_t = repeat(torch.tensor(sigmas[index], dtype=x.dtype), 
                         't -> b t l c', b=b, l=1, c=1).to(device)
        sqrt_one_minus_at = repeat(torch.tensor(sqrt_one_minus_alphas[index], dtype=x.dtype), 
                                   't -> b t l c', b=b, l=1, c=1).to(device)

        # current prediction for x_0
        if self.model.parameterization == "eps":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        elif self.model.parameterization == "v":
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        elif self.model.parameterization == "x0":
            pred_x0 = model_output
        else:
            raise NotImplementedError(self.model.parameterization)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.betas.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

if __name__ == "__main__":
    class Model(object):
        def __init__(self) -> None:
            self.num_timesteps = 50
    model = Model()
    ddim = DDIMSampler(model)

    total_n, base_n, ar_diff, template = 4, 4, 2, list(range(4, -1, -1))
    # step_matrix, step_index, step_mask  = ddim.generate_timestep_matrix(4, base_n, ar_diff, template)
    # print(step_index)
    step_matrix, step_index, step_update_mask, intervals = ddim.generate_timestep_matrix(total_n, base_n, ar_diff, template)
    for i in range(len(intervals)):
        print(intervals[i], step_update_mask[i], step_index[i])
    
    # print(step_index)
    # print(step_update_mask)
    # print(step_feed_mask)
