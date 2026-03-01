# .devcontainer/Dockerfile
# syntax=docker/dockerfile:1

FROM node:20

###############################################################################
# 0.  Basic packages + build prerequisites for Rust
###############################################################################
ARG TZ
ENV TZ=${TZ:-Etc/UTC}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config curl ca-certificates libssl-dev \
      less git procps sudo fzf zsh man-db unzip gnupg2 gh \
      iptables ipset iproute2 dnsutils aggregate jq \
   && rm -rf /var/lib/apt/lists/*

# Full passwordless sudo for node
RUN echo "node ALL=(root) NOPASSWD: ALL" > /etc/sudoers.d/yolo \
 && chmod 0440 /etc/sudoers.d/yolo

###############################################################################
# 1.  Rust toolchain (stable)
###############################################################################
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable \
 && /opt/cargo/bin/rustup component add clippy rustfmt
ENV PATH=${CARGO_HOME}/bin:$PATH
RUN chown -R node:node ${RUSTUP_HOME} ${CARGO_HOME}

###############################################################################
# 2.  Install Conda (no env creation)
###############################################################################
ARG MAMBAFORGE_VER=24.5.0-0
RUN curl -sL \
      "https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VER}/Mambaforge-${MAMBAFORGE_VER}-Linux-$(uname -m).sh" \
      -o mambaforge.sh \
 && bash mambaforge.sh -b -p /opt/conda \
 && rm mambaforge.sh \
 && chown -R node:node /opt/conda
ENV PATH=/opt/conda/bin:$PATH

###############################################################################
# 3.  Global npm path for Node user
###############################################################################
RUN mkdir -p /usr/local/share/npm-global && \
    chown -R node:node /usr/local/share

###############################################################################
# 4.  Shell history + devcontainer flag
###############################################################################
ARG USERNAME=node
RUN mkdir /commandhistory && touch /commandhistory/.bash_history && \
    chown -R ${USERNAME} /commandhistory
ENV DEVCONTAINER=true

###############################################################################
# 5.  Workspace, Claude config, delta
###############################################################################
RUN mkdir -p /workspace /home/node/.claude && \
    chown -R node:node /workspace /home/node/.claude
WORKDIR /workspace

RUN ARCH=$(dpkg --print-architecture) && \
    wget -q "https://github.com/dandavison/delta/releases/download/0.18.2/git-delta_0.18.2_${ARCH}.deb" && \
    dpkg -i "git-delta_0.18.2_${ARCH}.deb" && \
    rm "git-delta_0.18.2_${ARCH}.deb"

###############################################################################
# 6.  Zsh + Claude CLI
###############################################################################
USER node

RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate trading_env_f' >> ~/.zshrc \
 && echo 'export LD_LIBRARY_PATH=$(python -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / \"lib\")"):$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.zshrc

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin
ENV SHELL=/bin/zsh

RUN sh -c "$(wget -qO- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.0/zsh-in-docker.sh)" -- \
    -p git -p fzf \
    -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
    -a "source /usr/share/doc/fzf/examples/completion.zsh" \
    -a 'export PROMPT_COMMAND="history -a"' \
    -a 'export HISTFILE=/commandhistory/.bash_history' \
    -x

RUN npm install -g @anthropic-ai/claude-code

###############################################################################
# 7.  Firewall helper (unchanged)
###############################################################################
USER root
COPY init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
    chmod 0440 /etc/sudoers.d/node-firewall

###############################################################################
# 8.  Default to node user
###############################################################################
USER node
ENTRYPOINT ["zsh", "-l"]
