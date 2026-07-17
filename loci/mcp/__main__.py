"""Allow ``python -m loci.mcp`` as an alternative to the ``loci-mcp`` script."""

from loci.mcp.server import main

if __name__ == "__main__":
    main()
