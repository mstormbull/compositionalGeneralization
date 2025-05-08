# hash:sha256:ef1bf8ef9cc0589c8dd81daf21206eb3bcf112c699f92a8ea806746b52b60cd8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.10.1 \
    scikit-learn==1.6.1 \
    scipy==1.15.2
