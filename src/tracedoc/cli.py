"""Command-line interface."""

import typer

from .__about__ import NAME, __version__

app = typer.Typer(name=NAME, help="Diagnose LangGraph agent traces.", no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"{NAME} {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
