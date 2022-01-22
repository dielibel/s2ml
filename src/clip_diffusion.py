#@markdown # **CLIP-Guided Diffusion**
#@markdown ##### WARNING: This requires access to 16GB of VRAM reliably, so may not work for users not using Colab Pro/+
usingDiffusion = True;

torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()
use_gradient_checkpointing = True #@param {type:"boolean"}
#@markdown Enabling gradient checkpointing reduces VRAM memory use, but slows down computation.
prompt = "a burning chariot and a lost card"#@param {type:"string"}
batch_size = 1#@param {type:"number"}
#@markdown Note: x4 and x16 models for CLIP may not work reliably on lower-memory machines
clip_model = "ViT-B/32" #@param ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32','ViT-B/16']
    #@markdown Controls how much the image should look like the prompt.
clip_guidance_scale = 1000#@param {type:"number"}  
    #@markdown Controls the smoothness of the final output.
tv_scale = 150#@param {type:"number"}             
cutn = 32#@param {type:"number"}
cut_pow = 0.5#@param {type:"number"}
n_batches = 1#@param {type:"number"}
    #@markdown This can be an URL or Colab local path and must be in quotes.
init_image = "https://syrkis.ams3.cdn.digitaloceanspaces.com/noah/milton/2.jpg" #@param {type:"string"}
    #@markdown This needs to be between approx. 200 and 500 when using an init image.  
    #@markdown Higher values make the output look more like the init.
skip_timesteps = 0#@param {type:"number"}  
    

diffusion_steps = 1500#@param {type:"number"}

      

if seed == -1:
    seed = None

diff_image_size = 256 # size of image when using diffusion
diff_image_size = int(diff_image_size)

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': diffusion_steps,
    'rescale_timesteps': True,
    'timestep_respacing': str(diffusion_steps),  # Modify this value to decrease the number of
    'use_checkpoint': use_gradient_checkpointing,                            # timesteps.
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Executing using CLIP guided diffusion method')
if (prompt != None):
  print('Using prompt: '+ prompt)

print('Using device:', device)

model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(abs_root_path + "/models/" + '512x512_diffusion_uncond_finetune_008100.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
def do_run():
    if seed is not None:
        torch.manual_seed(seed)

    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            image_embeds = clip_model.encode_image(clip_in).float().view([cutn, n, -1])
            dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            return -torch.autograd.grad(loss, x)[0]

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        
        for j, sample in enumerate(samples):
            cur_t -= 1     
            for k, image in enumerate(sample['pred_xstart']):
                filename = f'diffusion-steps/{batch_size * j:05}.png'
                TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                if j % display_frequency == 0 or cur_t == -1:
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    print()      
                    display.display(display.Image(filename))
            
                

do_run()