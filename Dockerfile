FROM python:3.7

MAINTAINER dielibel@syrkis.com

WORKDIR /usr/src/app

COPY . .

RUN \
    apt update && apt install -y exempi && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

RUN \
    mkdir models vqgan-steps diffusion-steps && \
    curl -L -o models/wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' && \
    curl -L -o models/wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml'

CMD ls
