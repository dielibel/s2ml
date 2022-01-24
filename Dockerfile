FROM python:3.7

MAINTAINER dielibel@syrkis.com

WORKDIR /usr/src/app

COPY . .

RUN \
    git clone https://github.com/openai/CLIP && \
    git clone https://github.com/dielibel/guided-diffusion && \
    python -m pip install --upgrade pip && \
    pip install -e ./CLIP && \
    pip install -e ./guided-diffusion && \
    git clone https://github.com/CompVis/taming-transformers && \
    pip install ftfy regex tqdm omegaconf pytorch-lightning && \
    pip install kornia && \
    pip install einops && \
    pip install transformers && \
    pip install stegano && \
    apt update && apt install -y exempi && \
    pip install python-xmp-toolkit && \
    pip install imgtag && \
    pip install pillow==7.1.2 && \
    pip install taming-transformers && \
    git clone https://github.com/xinntao/ESRGAN && \
    pip install imageio-ffmpeg

RUN \
    mkdir models && \
    curl -L -o models/vqgan_imagenet_f16_1024.ckpt -C - 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' && \
    curl -L -o models/vqgan_imagenet_f16_1024.yaml -C - 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' && \
    curl -L -o models/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' && \
    curl -L -o models/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' && \
    curl -L -o models/coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml' && \
    curl -L -o models/coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt' && \
    curl -L -o models/wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' && \
    curl -L -o models/wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml '&& \
    curl -L -o models/sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' && \
    curl -L -o models/sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' && \
    curl -L -o models/512x512_diffusion_uncond_finetune_008100.pt -C - 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt' &&\
    curl -L -o models/256x256_diffusion_uncond_finetune_008100.pt -C - 'https://v-diffusion.s3.us-west-2.amazonaws.com/256x256_diffusion_uncond_finetune_008100.pt' 

ENTRYPOINT ["python", "main.py"]

CMD ["no_config_specified"]
