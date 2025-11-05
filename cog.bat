@echo off
py -3.11 -c "import sys; from cog.cli import cli; sys.exit(cli())" %*
