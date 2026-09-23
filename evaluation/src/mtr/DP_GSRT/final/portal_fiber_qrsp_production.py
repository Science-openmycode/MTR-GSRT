"""Production-only entry point for authenticated portal-fiber QRSP decoding."""
from __future__ import annotations

import sys

import quotient_rsp_bridge_development as decoder


def main() -> None:
    if "--production-postprocess" not in sys.argv:
        raise RuntimeError("Production QRSP requires --production-postprocess")
    if "--acknowledge-non-dp-development" in sys.argv:
        raise RuntimeError("Development mode is forbidden through the production entry point")
    decoder.main()


if __name__ == "__main__":
    main()
