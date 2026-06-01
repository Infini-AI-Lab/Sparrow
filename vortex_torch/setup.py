import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def _normalize_single_arch(arch: str):
    """
    Normalize a single architecture string into (major, minor, wants_ptx).

    Supported examples:
    - "8.9"
    - "9.0+PTX"
    - "89"
    - "sm_89"
    - "compute_89"
    """
    arch = arch.strip().lower()
    wants_ptx = arch.endswith("+ptx")

    if wants_ptx:
        arch = arch[:-4]

    if arch.startswith("sm_"):
        arch = arch[3:]
    elif arch.startswith("compute_"):
        arch = arch[len("compute_"):]

    if "." in arch:
        major, minor = arch.split(".", 1)
        return int(major), int(minor), wants_ptx

    if arch.isdigit():
        if len(arch) == 2:
            return int(arch[0]), int(arch[1]), wants_ptx
        elif len(arch) == 3:
            return int(arch[:-1]), int(arch[-1]), wants_ptx

    raise ValueError(f"Unsupported CUDA arch format: {arch}")


def get_cuda_gencode_flags(include_ptx: bool = False):
    """
    Generate NVCC -gencode flags automatically.

    Priority:
    1. TORCH_CUDA_ARCH_LIST environment variable
    2. Detect capabilities from visible local GPUs via torch.cuda

    Args:
        include_ptx: Whether to also emit PTX fallback flags.

    Returns:
        A list of NVCC flags such as:
        [
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ]
    """
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    capabilities = set()
    ptx_capabilities = set()

    if arch_list_env:
        # Example values:
        # "8.9 9.0"
        # "8.9+PTX"
        # "8.9;9.0"
        # "sm_89 sm_90"
        raw_items = arch_list_env.replace(";", " ").split()
        for item in raw_items:
            major, minor, wants_ptx = _normalize_single_arch(item)
            capabilities.add((major, minor))
            if wants_ptx:
                ptx_capabilities.add((major, minor))
    else:
        # Import torch lazily so the file can still be imported in some environments.
        import torch

        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(device_idx)
                capabilities.add((major, minor))

    flags = []
    for major, minor in sorted(capabilities):
        compute = f"compute_{major}{minor}"
        sm = f"sm_{major}{minor}"
        flags.append(f"-gencode=arch={compute},code={sm}")

        if include_ptx or (major, minor) in ptx_capabilities:
            flags.append(f"-gencode=arch={compute},code={compute}")

    return flags


nvcc_flags = [
    "-O3",
] + get_cuda_gencode_flags(include_ptx=False)


setup(
    name="vortex_torch",
    version="0.3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7",
        "lighteval[math]==0.12.2",
        "huggingface_hub==0.36.2",
        "inspect-ai==0.3.207",
        "transformers==4.57.6",
        "s3fs==2025.9.0"
    ],
    ext_modules=[
        CUDAExtension(
            name="vortex_torch_C",
            sources=[
                "csrc/register.cc",
                "csrc/utils_sglang.cu",
                "csrc/utils_sglang_v2.cu",
                "csrc/topk.cu",
                "csrc/topk_v2.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
