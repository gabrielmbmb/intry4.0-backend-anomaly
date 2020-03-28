ARG PYTHON_VERSION=3.7-slim
FROM python:${PYTHON_VERSION}

LABEL maintainer="gmartin_b@usal.es"

RUN apt-get update && \
    pip install --upgrade pip && \
    useradd --create-home worker

USER worker
WORKDIR /home/worker

# Install requirements first to leverage Docker cache
ADD --chown=worker:worker requirements.txt blackbox/requirements.txt
RUN python -m pip install --user --no-warn-script-location -r blackbox/requirements.txt

# Install the app
ADD --chown=worker:worker . blackbox/
WORKDIR "/home/worker/blackbox/"

ENV PATH="/home/worker/.local/bin:${PATH}"

CMD ["gunicorn", "--bind", "0.0.0.0:5678", "-w", "4", "blackbox.app.app:create_app()"]