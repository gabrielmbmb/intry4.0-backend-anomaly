ARG PYTHON_VERSION=3.7-slim
FROM python:${PYTHON_VERSION}

LABEL maintainer="gmartin_b@usal.es"

RUN apt-get update && \
    pip install --upgrade pip && \
    useradd --create-home worker

USER worker
WORKDIR /home/worker

# install requirements first to leverage Docker cache
ADD --chown=worker:worker requirements.txt blackbox/requirements.txt
RUN python -m pip install --user -r blackbox/requirements.txt

# install the app
ADD --chown=worker:worker . blackbox/
RUN cd blackbox && \
    python -m pip install --user .

ENV PATH="/home/worker/.local/bin:${PATH}"

CMD blackbox