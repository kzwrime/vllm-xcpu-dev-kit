#!/usr/bin/env bash
set -euo pipefail

MOONCAKE_ROOT="${MOONCAKE_ROOT:-/home/wuzhikun/vllm-xcpu-dev-kit-0.19/Mooncake}"
DEPS_DIR="${DEPS_DIR:-${MOONCAKE_ROOT}/deps}"
DEPS_PREFIX="${DEPS_PREFIX:-${DEPS_DIR}/_install}"
ENV_FILE="${ENV_FILE:-${DEPS_DIR}/env.sh}"
BUILD_DIR_NAME="${BUILD_DIR_NAME:-build-sw64}"
JOBS="${JOBS:-$(nproc)}"
BUILD_MOONCAKE="${BUILD_MOONCAKE:-1}"
CLEAN="${CLEAN:-1}"
LOG_FILE="${LOG_FILE:-${MOONCAKE_ROOT}/build/sw64_rebuild_deps.log}"

GFLAGS_REF="${GFLAGS_REF:-v2.2.2}"
GLOG_REF="${GLOG_REF:-v0.6.0}"
YLT_REF="${YLT_REF:-6a0e067d9a43492cf8e4e280b531924fbd724dbd}"
YAML_CPP_REF="${YAML_CPP_REF:-0.8.0}"
MSGPACK_REF="${MSGPACK_REF:-cpp-6.1.0}"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.12}"
PYTHON_LIB="${PYTHON_LIB:-/usr/local/lib/libpython3.12.so}"
PYTHON_INCLUDE="${PYTHON_INCLUDE:-/usr/local/include/python3.12}"

if [ -n "${LOG_FILE}" ]; then
    mkdir -p "$(dirname "${LOG_FILE}")"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

export PATH="${DEPS_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:${LIBRARY_PATH:-}"
export CPATH="${DEPS_PREFIX}/include:${CPATH:-}"
export PKG_CONFIG_PATH="${DEPS_PREFIX}/lib64/pkgconfig:${DEPS_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${DEPS_PREFIX}:${CMAKE_PREFIX_PATH:-}"

write_env_file() {
    cat >"${ENV_FILE}" <<EOF
# Source this file before running Mooncake binaries built with private deps.
export MOONCAKE_ROOT="${MOONCAKE_ROOT}"
export DEPS_DIR="${DEPS_DIR}"
export DEPS_PREFIX="${DEPS_PREFIX}"
export PATH="${DEPS_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${DEPS_PREFIX}/lib64:${DEPS_PREFIX}/lib:\${LIBRARY_PATH:-}"
export CPATH="${DEPS_PREFIX}/include:\${CPATH:-}"
export PKG_CONFIG_PATH="${DEPS_PREFIX}/lib64/pkgconfig:${DEPS_PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${DEPS_PREFIX}:\${CMAKE_PREFIX_PATH:-}"
EOF
}

log() {
    printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

checkout_repo() {
    local name="$1"
    local url="$2"
    local ref="$3"

    log "Preparing ${name} at ${ref}"
    if [ ! -d "${DEPS_DIR}/${name}/.git" ]; then
        git clone "${url}" "${DEPS_DIR}/${name}"
    fi

    if ! git -C "${DEPS_DIR}/${name}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
        git -C "${DEPS_DIR}/${name}" fetch --tags origin
    fi
    git -C "${DEPS_DIR}/${name}" checkout "${ref}"
}

cmake_pkg_dir() {
    local pattern="$1"
    local found

    found="$(find "${DEPS_PREFIX}" -type f -name "${pattern}" -print -quit)"
    if [ -z "${found}" ]; then
        printf 'Could not find %s under %s\n' "${pattern}" "${DEPS_PREFIX}" >&2
        return 1
    fi

    dirname "${found}"
}

build_and_install() {
    local name="$1"
    shift

    log "Configuring ${name}"
    cmake -S "${DEPS_DIR}/${name}" -B "${DEPS_DIR}/${name}/${BUILD_DIR_NAME}" "$@"

    log "Building ${name} with -j${JOBS}"
    cmake --build "${DEPS_DIR}/${name}/${BUILD_DIR_NAME}" -j"${JOBS}"

    log "Installing ${name} to ${DEPS_PREFIX}"
    cmake --install "${DEPS_DIR}/${name}/${BUILD_DIR_NAME}"
}

mkdir -p "${DEPS_DIR}" "${DEPS_PREFIX}"
write_env_file

if [ "${CLEAN}" = "1" ]; then
    log "Cleaning old build directories and private install prefix"
    rm -rf "${MOONCAKE_ROOT}/build" "${DEPS_PREFIX}"
    find "${DEPS_DIR}" -mindepth 2 -maxdepth 2 -type d \( \
        -name build -o \
        -name "${BUILD_DIR_NAME}" \
    \) -print -exec rm -rf {} +
    mkdir -p "${DEPS_PREFIX}"
fi

checkout_repo gflags https://github.com/gflags/gflags.git "${GFLAGS_REF}"
build_and_install gflags \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_gflags_LIB=ON \
    -DBUILD_gflags_nothreads_LIB=ON

checkout_repo glog https://github.com/google/glog.git "${GLOG_REF}"
build_and_install glog \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_GFLAGS=ON \
    -Dgflags_DIR="$(cmake_pkg_dir gflags-config.cmake)"

checkout_repo yalantinglibs https://github.com/alibaba/yalantinglibs.git "${YLT_REF}"
build_and_install yalantinglibs \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_UNIT_TESTS=OFF

checkout_repo yaml-cpp https://github.com/jbeder/yaml-cpp.git "${YAML_CPP_REF}"
build_and_install yaml-cpp \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF \
    -DYAML_CPP_BUILD_CONTRIB=OFF

checkout_repo msgpack-c https://github.com/msgpack/msgpack-c.git "${MSGPACK_REF}"
build_and_install msgpack-c \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${DEPS_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON

log "Installed dependency package files"
find "${DEPS_PREFIX}" -type f \( \
    -name 'gflags-config.cmake' -o \
    -name 'glog-config.cmake' -o \
    -name 'yalantinglibsConfig.cmake' -o \
    -name 'yaml-cpp-config.cmake' -o \
    -name 'msgpack*.cmake' \
\) -print | sort

log "Configuring Mooncake"
cd "${MOONCAKE_ROOT}"
git submodule update --init --recursive

cmake -S . -B build \
    -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
    -DPYTHON_LIBRARY="${PYTHON_LIB}" \
    -DPYTHON_INCLUDE_DIR="${PYTHON_INCLUDE}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    -DPython3_LIBRARY="${PYTHON_LIB}" \
    -DPython3_INCLUDE_DIR="${PYTHON_INCLUDE}" \
    -Dgflags_DIR="$(cmake_pkg_dir gflags-config.cmake)" \
    -Dglog_DIR="$(cmake_pkg_dir glog-config.cmake)" \
    -Dyalantinglibs_DIR="$(cmake_pkg_dir yalantinglibsConfig.cmake)" \
    -Dyaml-cpp_DIR="$(cmake_pkg_dir yaml-cpp-config.cmake)" \
    -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}"

if [ "${BUILD_MOONCAKE}" = "1" ]; then
    log "Building Mooncake with -j${JOBS}"
    cmake --build build -j"${JOBS}"
else
    log "Skipping Mooncake build because BUILD_MOONCAKE=${BUILD_MOONCAKE}"
fi

log "Done"
log "Runtime environment file: ${ENV_FILE}"
