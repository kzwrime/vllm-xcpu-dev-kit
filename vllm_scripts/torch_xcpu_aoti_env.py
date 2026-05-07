#!/usr/bin/env python3
import importlib.util
import os
import shlex
from pathlib import Path


def emit_export(name: str, value: str | Path) -> None:
    print(f"export {name}={shlex.quote(str(value))}")


def find_package_aoti_bits(package: str, header_name: str) -> tuple[Path, Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{package} not found in current Python env")

    install_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    header = install_dir / "include" / header_name
    so_files = sorted(install_dir.glob("_C*.so"))

    if not header.is_file():
        raise SystemExit(f"missing header: {header}")
    if not so_files:
        raise SystemExit(f"missing {package} extension: {install_dir}/_C*.so")

    return header, so_files[0]


def main() -> None:
    packages = (
        ("torch_xcpu", "aoti_torch_xcpu.h"),
        ("torch_mpi_ext", "aoti_torch_mpi_ext.h"),
    )

    headers: list[Path] = []
    so_files: list[Path] = []
    for package, header_name in packages:
        header, so_file = find_package_aoti_bits(package, header_name)
        headers.append(header)
        so_files.append(so_file)

    aoti_extra_cflags = " ".join(f"-include {header}" for header in headers)
    aoti_extra_ldflags = " ".join(
        f"-Wl,-rpath,{so_file.parent} {so_file}" for so_file in so_files
    )

    if os.environ.get("AOTI_EXTRA_CFLAGS"):
        aoti_extra_cflags += f" {os.environ['AOTI_EXTRA_CFLAGS']}"
    if os.environ.get("AOTI_EXTRA_LDFLAGS"):
        aoti_extra_ldflags += f" {os.environ['AOTI_EXTRA_LDFLAGS']}"

    emit_export("AOTI_EXTRA_CFLAGS", aoti_extra_cflags)
    emit_export("AOTI_EXTRA_LDFLAGS", aoti_extra_ldflags)


if __name__ == "__main__":
    main()
