import typer


def entrypoint(person: str):
    print(f"Hello, {person}!")


if __name__ == "__main__":
    typer.run(entrypoint)