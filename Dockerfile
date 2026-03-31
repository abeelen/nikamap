# Dockerfile multi-python pour CI tox nikamap
FROM debian:stable-slim

# Installer les dépendances système pour pyenv et Python build
RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installer pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

# Installer les versions Python nécessaires
RUN pyenv install 3.10 && \
    pyenv install 3.11 && \
    pyenv install 3.12 && \
    pyenv install 3.13

# Ajouter les shims pyenv au PATH pour tox
ENV PATH="$PYENV_ROOT/shims:$PATH"

# Installer pip et tox globalement (pour tous les Pythons)
RUN pyenv global 3.10 3.11 3.12 3.13 && \
    pip install --upgrade pip && \
    pip install tox

# Par défaut, lance bash (pour debug)
CMD ["bash"]
