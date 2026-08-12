FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc g++ cmake ninja-build make \
        python3 python3-dev python3.12-venv \
        libopenblas-dev libnuma-dev \
        git ssh openssh-server gpg \
        wget curl net-tools \
        vim tmux tree \
        zip unzip xz-utils \
        openmpi-bin libopenmpi-dev \
        libtcmalloc-minimal4 \
        gdb \
    && rm -rf /var/lib/apt/lists/*

RUN printf '%s\n' \
      'PermitRootLogin prohibit-password' \
      'PubkeyAuthentication yes' \
      'PasswordAuthentication no' \
      'KbdInteractiveAuthentication no' \
      'UsePAM no' \
      > /etc/ssh/sshd_config.d/10-cluster.conf \
    && printf '%s\n' \
      'Host node01 node02 node03 node04 node05 node06 node07 node08' \
      '  StrictHostKeyChecking accept-new' \
      > /etc/ssh/ssh_config.d/10-cluster.conf \
    && printf '%s\n' \
      '# Enable OpenMPI root access' \
      'export OMPI_ALLOW_RUN_AS_ROOT=1' \
      'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1' \
      >> /root/.bashrc

COPY entrypoint.sh /usr/local/sbin/cluster-entrypoint
RUN chmod 0755 /usr/local/sbin/cluster-entrypoint

ENTRYPOINT ["/usr/local/sbin/cluster-entrypoint"]
