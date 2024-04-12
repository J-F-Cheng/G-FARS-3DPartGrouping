import torch
import time
import tqdm
from torch_scatter import scatter_sum

def samples_gen(network, x, emb_pcs, eps, sampler="PC_origin", num_steps=500):
    conf = network.conf
    x_states = []
    pc_final_cor_steps = []
    
    if sampler == "PC_origin":
        t = torch.ones(emb_pcs.size(0), device=conf.device) * conf.t0
        x.x *= network.margin_fn(t)[:, None]
        time_steps = torch.linspace(conf.t0, eps, num_steps, device=conf.device)
        step_size = time_steps[0] - time_steps[1]
        with torch.no_grad():
            iter = 0
            t0_time = time.time()
            t0_perf_counter = time.perf_counter()
            t0_process_time = time.process_time()
            for time_step in tqdm.tqdm(time_steps):
                batch_time_step = torch.ones(emb_pcs.size(0), device=conf.device) * time_step
                batch_time_step = batch_time_step.unsqueeze(-1)

                # Corrector step (Langevin MCMC)
                grad = network(x_pose=x, t=batch_time_step, emb_pcs=emb_pcs)
                grad_norm = torch.square(grad).sum(dim=-1)
                grad_norm = torch.sqrt(scatter_sum(grad_norm, x.batch, dim=0))[x.batch].unsqueeze(-1)
                noise_norm = torch.sqrt(
                    scatter_sum(torch.ones(x.x.size(0), device=conf.device) * network.input_dim, x.batch, dim=0))
                noise_norm = noise_norm.unsqueeze(-1)
                noise_norm = noise_norm[x.batch]
                langevin_step_size = 2 * (conf.snr * noise_norm / grad_norm) ** 2
                x.x = x.x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x.x)

                # Predictor step (Euler-Maruyama)
                g = network.diffusion_coeff_fn(batch_time_step)
                x_mean = x.x + (g ** 2) * network(x_pose=x, t=batch_time_step, emb_pcs=emb_pcs) * step_size
                x.x = x_mean + torch.sqrt(g ** 2 * step_size) * torch.randn_like(x.x)
                x_states.append(x_mean)
                iter += 1
            samp_time = {"time": time.time() - t0_time, "perf_counter": time.perf_counter() - t0_perf_counter, "process_time": time.process_time() - t0_process_time}
            pc_final_cor_steps.append(x_mean)
    
    elif sampler == "EM":
        t = torch.ones(emb_pcs.size(0), device=conf.device) * conf.t0
        x.x *= network.margin_fn(t)[:, None]
        time_steps = torch.linspace(conf.t0, eps, num_steps, device=conf.device)
        step_size = time_steps[0] - time_steps[1]
        with torch.no_grad():
            iter = 0
            t0_time = time.time()
            t0_perf_counter = time.perf_counter()
            t0_process_time = time.process_time()
            for time_step in tqdm.tqdm(time_steps):
                batch_time_step = torch.ones(emb_pcs.size(0), device=conf.device) * time_step
                batch_time_step = batch_time_step.unsqueeze(-1)

                # Predictor step (Euler-Maruyama)
                g = network.diffusion_coeff_fn(batch_time_step)
                x_mean = x.x + (g ** 2) * network(x_pose=x, t=batch_time_step, emb_pcs=emb_pcs) * step_size
                x.x = x_mean + torch.sqrt(g ** 2 * step_size) * torch.randn_like(x.x)
                x_states.append(x_mean)
                iter += 1
            samp_time = {"time": time.time() - t0_time, "perf_counter": time.perf_counter() - t0_perf_counter, "process_time": time.process_time() - t0_process_time}
            pc_final_cor_steps.append(x_mean)

    else:
        print("Not a recognised sampler!")
        raise (NotImplementedError)

    return {"x_states": x_states, "pred_part_poses": x_mean, "pc_final_cor_steps": pc_final_cor_steps, "samp_time": samp_time}
