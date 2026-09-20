"""Commands for decisions, serving, and benchmarks."""

import json
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn

from system1.backend import GemmaBackend
from system1.bench import benchmark, report_table
from system1.core import SystemOne
from system1.data import download_typed_decisions
from system1.schema import Request

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def fail(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(1)


@app.command()
def ask(
    state: Annotated[str, typer.Option()],
    choice: Annotated[list[str] | None, typer.Option()] = None,
    noul: Annotated[list[str] | None, typer.Option()] = None,
    score: Annotated[list[str] | None, typer.Option()] = None,
    permutations: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    """Print answer JSON for named questions."""
    backend = GemmaBackend()
    try:
        questions = {}
        for kind, entries in (("choice", choice), ("score", score), ("noul", noul)):
            for entry in entries or []:
                name, separator, value = entry.partition(":")
                instructions = name
                if kind != "noul" and "=" in name:
                    name, instructions = name.split("=", 1)
                    if not instructions.strip():
                        raise ValueError("Question instructions must not be empty")
                name = name.strip()
                if not separator or not name or not value.strip() or name in questions:
                    raise ValueError("Use unique name:value questions")
                question = {
                    "type": kind,
                    "instructions": value if kind == "noul" else instructions.strip(),
                }
                if kind != "noul":
                    labels = [label.strip() for label in value.split(",")]
                    if len(set(labels)) != len(labels):
                        raise ValueError("Option labels must be unique")
                    question["criteria" if kind == "choice" else "levels"] = (
                        dict.fromkeys(labels) if kind == "choice" else labels
                    )
                questions[name] = question
        parsed_state = (
            json.loads(state) if state.lstrip().startswith(("{", "[")) else state
        )
        request = Request(
            state=parsed_state, questions=questions, permutations=permutations
        )
        typer.echo(
            json.dumps(SystemOne(backend).decide(request), indent=2, allow_nan=False)
        )
    except (OSError, ValueError, httpx.HTTPError, KeyError, IndexError) as error:
        fail(error)
    finally:
        backend.close()


@app.command()
def serve(port: Annotated[int, typer.Option(min=1, max=65535)] = 8020) -> None:
    """Serve the decision API on localhost."""
    uvicorn.run("system1.api:create_app", factory=True, host="127.0.0.1", port=port)


@app.command()
def fetch_typed_decisions(
    out_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Convert local typed decisions parquet files to benchmark JSONL."""
    try:
        for path in download_typed_decisions(out_dir):
            typer.echo(str(path))
    except ImportError:
        fail(ValueError("Install benchmark dependencies with uv sync --extra bench"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        fail(error)


@app.command()
def bench(
    cases: Path,
    fit_temperature: Annotated[Path | None, typer.Option()] = None,
    permutations: Annotated[int, typer.Option(min=1)] = 3,
    out: Annotated[Path, typer.Option()] = Path("report.json"),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workflow: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Evaluate JSONL cases and optionally fit on separate training cases."""
    backend = GemmaBackend()
    try:
        report = benchmark(
            cases,
            SystemOne(backend),
            permutations,
            fit_temperature,
            limit=limit,
            workflow=workflow,
        )
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        typer.echo(report_table(report))
    except (OSError, ValueError, httpx.HTTPError, KeyError, IndexError) as error:
        fail(error)
    finally:
        backend.close()
