# Third-party source checkouts

This directory is populated by pinned Git submodules:

- vggt-omega: primary geometry backbone; FAIR Noncommercial Research License.
- sam3: optional SAM 3.1 segmentation/tracking teacher and baseline; SAM License.
- videomt: optional video query propagation reference; MIT License.

Initialize them with:

    git submodule update --init --recursive

Do not edit upstream files directly. Keep any E3A-required patch under patches/<upstream>/, pin the upstream commit, and add parity tests. Do not commit upstream checkpoints, Hugging Face tokens, or dataset files.
