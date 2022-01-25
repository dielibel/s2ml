FROM python:3.7

WORKDIR /usr/src/app

RUN apt update && apt install -y exempi

RUN \
    mkdir models vqgan-steps diffusion-steps && \
    curl -L -o models/wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' && \
    curl -L -o models/wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml'

COPY requirements.txt ESRGAN taming-transformers ./

RUN pip install -r requirements.txt

COPY script.py ./

CMD python script.py
