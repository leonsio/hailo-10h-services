Place exactly one matching HailoRT **userspace** `.deb` and its CPython 3.13
GenAI `.whl` here before building. Do not put driver/DKMS packages here.
For ARM64 HailoRT 5.4.0:

- `hailort_5.4.0_arm64.deb`
- `hailort-5.4.0-cp313-cp313-linux_aarch64.whl`

The host kernel driver and device firmware must match. Proprietary packages
are supplied locally and are not distributed in this repository.
