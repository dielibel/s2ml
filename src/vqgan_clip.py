def vqgan_clip(args):

    usingDiffusion = False;

    prompts = args.prompts #@param {type:"string"}
    width =  args.width #@param {type:"number"}
    height =  args.height#@param {type:"number"}
    #@markdown Note: x4 and x16 models for CLIP may not work reliably on lower-memory machines
    clip_model = args.clip_model #@param ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32','ViT-B/16']
    vqgan_model = args.vqgan_model #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "sflckr"]
    initial_image = args.initial_image #@param {type:"string"}
    target_images = args.target_images #@param {type:"string"}
    max_iterations = args.max_iterations #@param {type:"number"}
    print(8)
    exit()

    input_images = ""
    #@markdown ### Advanced VQGAN+CLIP Parameters 

    vq_init_weight = 0.0#@param {type:"number"}
    vq_step_size = 0.1#@param {type:"number"}
    vq_cutn = 64#@param {type:"number"}
    vq_cutpow = 1.0#@param {type:"number"}

    model_names={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 
                     "wikiart_1024":"WikiArt 1024", "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR"}
    model_name = model_names[vqgan_model]     

    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()

    if seed == -1:
        seed = None
    if initial_image == "None":
        initial_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    if initial_image or target_images != []:
        input_images = True

    prompts = [frase.strip() for frase in prompts.split("|")]
    if prompts == ['']:
        prompts = []



    args = argparse.Namespace(
            prompts=prompts,
            image_prompts=target_images,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[width, height],
            init_image=initial_image,
            init_weight=0.,
            clip_model= clip_model,
            vqgan_config=f'models/{vqgan_model}.yaml',
            vqgan_checkpoint=f'models/{vqgan_model}.ckpt',
            step_size=vq_step_size,
            cutn=vq_cutn,
            cut_pow=vq_cutpow,
            display_freq=display_frequency,
            seed=seed,
        )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    notebook_name = "VQGAN+CLIP"
    print('Executing using VQGAN+CLIP method')
    print('Using device:', device)
    if prompts:
        print('Using text prompt:', prompts)
    if target_images:
        print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


    if args.init_image:
        pil_image = Image.open(args.init_image).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    def add_xmp_data(nombrefichero):
        image = ImgTag(filename=nombrefichero)
        image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'creator', 'VQGAN+CLIP', {"prop_array_is_ordered":True, "prop_value_is_array":True})
        if args.prompts:
            image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', " | ".join(args.prompts), {"prop_array_is_ordered":True, "prop_value_is_array":True})
        else:
            image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', 'None', {"prop_array_is_ordered":True, "prop_value_is_array":True})
        image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'i', str(i), {"prop_array_is_ordered":True, "prop_value_is_array":True})
        image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'model', model_name, {"prop_array_is_ordered":True, "prop_value_is_array":True})
        image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'seed',str(seed) , {"prop_array_is_ordered":True, "prop_value_is_array":True})
        image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'input_images',str(input_images) , {"prop_array_is_ordered":True, "prop_value_is_array":True})
        #for frases in args.prompts:
        #    image.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'Prompt' ,frases, {"prop_array_is_ordered":True, "prop_value_is_array":True})
        image.close()

    def add_stegano_data(filename):
        data = {
            "title": " | ".join(args.prompts) if args.prompts else None,
            "notebook": notebook_name,
            "i": i,
            "model": model_name,
            "seed": str(seed),
            "input_images": input_images
        }
        lsb.hide(filename, json.dumps(data)).save(filename)

    @torch.no_grad()
    def checkin(i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        add_stegano_data('progress.png')
        add_xmp_data('progress.png')
        display.display(display.Image('progress.png'))

    def ascend_txt():
        global i
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        filename = f"vqgan-steps/{i:04}.png"
        imageio.imwrite(filename, np.array(img))
        add_stegano_data(filename)
        add_xmp_data(filename)
        return result

    def train(i):
        opt.zero_grad()
        lossAll = ascend_txt()
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    try:
        with tqdm() as pbar:
            while True:
                train(i)
                if i == max_iterations:
                    break
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass
