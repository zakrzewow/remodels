"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """remodels."""


if __name__ == "__main__":
    main(prog_name="remodels")  # pragma: no cover
