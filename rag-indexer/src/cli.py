"""RAG Indexer CLI 인터페이스.

소스 관리, 동기화 트리거, 인덱싱된 콘텐츠 검색 명령어 제공.

종료 코드:
    0: 성공
    1: 일반/입력 오류 (잘못된 인자, 필수 값 누락)
    2: 설정 오류 (API 키 누락, 잘못된 설정)
    3: 연결 오류 (Qdrant, Notion API, 네트워크 문제)
    4: 동기화/처리 오류 (부분 실패, 데이터 문제)
    5: 내부 오류 (예상치 못한 예외)
"""

from datetime import datetime
from enum import IntEnum
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_settings
from .connectors import NotionConnector, SwaggerConnector
from .logging_config import configure_logging
from .models import NotionSourceConfig, Source, SourceType, SwaggerSourceConfig
from .scheduler import get_scheduler
from .services import get_indexer
from .storage import get_storage


class ExitCode(IntEnum):
    """CLI 작업을 위한 표준화된 종료 코드."""

    SUCCESS = 0  # 성공
    INPUT_ERROR = 1  # 잘못된 인자, 필수 값 누락
    CONFIG_ERROR = 2  # API 키 누락, 잘못된 설정
    CONNECTION_ERROR = 3  # Qdrant, Notion API, 네트워크 문제
    SYNC_ERROR = 4  # 부분 실패, 데이터 처리 문제
    INTERNAL_ERROR = 5  # 예상치 못한 예외

# Initialize CLI app
app = typer.Typer(
    name="rag-indexer",
    help="RAG Document Indexer - Sync and index documents from Notion and Swagger",
    add_completion=False,
)

# Sub-apps for command groups
source_app = typer.Typer(help="Manage data sources")
sync_app = typer.Typer(help="Sync operations")
scheduler_app = typer.Typer(help="Scheduler operations")

app.add_typer(source_app, name="source")
app.add_typer(sync_app, name="sync")
app.add_typer(scheduler_app, name="scheduler")

# Rich console for output
console = Console()


# ==================== Source Commands ====================


@source_app.command("list")
def source_list():
    """List all registered sources."""
    storage = get_storage()
    sources = storage.get_sources()

    if not sources:
        console.print("[yellow]No sources registered.[/yellow]")
        return

    table = Table(title="Registered Sources")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Active", style="blue")
    table.add_column("Last Synced", style="yellow")

    for source in sources:
        last_synced = source.last_synced.strftime("%Y-%m-%d %H:%M") if source.last_synced else "Never"
        table.add_row(
            source.id[:8],
            source.name,
            source.source_type.value,
            "✓" if source.is_active else "✗",
            last_synced,
        )

    console.print(table)


@source_app.command("add")
def source_add(
    name: str = typer.Option(..., "--name", "-n", help="Source name"),
    source_type: str = typer.Option(..., "--type", "-t", help="Source type (notion/swagger)"),
    # Notion options
    page_ids: Optional[str] = typer.Option(None, "--page-ids", help="Comma-separated Notion page IDs"),
    database_ids: Optional[str] = typer.Option(None, "--database-ids", help="Comma-separated Notion database IDs"),
    # Swagger options
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Swagger/OpenAPI URL or file path"),
):
    """Add a new data source."""
    storage = get_storage()

    # 소스 타입 검증
    try:
        src_type = SourceType(source_type.lower())
    except ValueError:
        console.print(f"[red]Invalid source type: {source_type}. Use 'notion' or 'swagger'.[/red]")
        raise typer.Exit(ExitCode.INPUT_ERROR)

    # 타입별 설정 구성
    if src_type == SourceType.NOTION:
        if not page_ids and not database_ids:
            console.print("[red]For Notion sources, provide --page-ids and/or --database-ids.[/red]")
            raise typer.Exit(ExitCode.INPUT_ERROR)

        config = NotionSourceConfig(
            page_ids=[p.strip() for p in (page_ids or "").split(",") if p.strip()],
            database_ids=[d.strip() for d in (database_ids or "").split(",") if d.strip()],
        )
    elif src_type == SourceType.SWAGGER:
        if not url:
            console.print("[red]For Swagger sources, provide --url.[/red]")
            raise typer.Exit(ExitCode.INPUT_ERROR)

        config = SwaggerSourceConfig(url=url)
    else:
        console.print(f"[red]Unsupported source type: {src_type}[/red]")
        raise typer.Exit(ExitCode.INPUT_ERROR)

    # 소스 생성 및 저장
    source = Source(
        name=name,
        source_type=src_type,
        config=config,
    )

    try:
        storage.add_source(source)
        console.print(f"[green]✓ Source '{name}' added successfully![/green]")
        console.print(f"  ID: {source.id}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(ExitCode.INPUT_ERROR)


@source_app.command("remove")
def source_remove(
    source_id: str = typer.Argument(..., help="Source ID to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a data source."""
    storage = get_storage()

    source = storage.get_source(source_id)
    if not source:
        # Try to find by name
        source = storage.get_source_by_name(source_id)

    if not source:
        console.print(f"[red]Source not found: {source_id}[/red]")
        raise typer.Exit(ExitCode.INPUT_ERROR)

    if not force:
        confirm = typer.confirm(f"Remove source '{source.name}'? This will delete all indexed documents.")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(ExitCode.SUCCESS)

    # Delete from vector store
    indexer = get_indexer()
    indexer.delete_source_chunks(source.id)

    # Delete from storage
    storage.delete_source(source.id)
    console.print(f"[green]✓ Source '{source.name}' removed.[/green]")


@source_app.command("show")
def source_show(
    source_id: str = typer.Argument(..., help="Source ID to show"),
):
    """Show source details."""
    storage = get_storage()

    source = storage.get_source(source_id)
    if not source:
        source = storage.get_source_by_name(source_id)

    if not source:
        console.print(f"[red]Source not found: {source_id}[/red]")
        raise typer.Exit(ExitCode.INPUT_ERROR)

    # 문서 수 조회
    docs = storage.get_documents(source.id)

    panel_content = f"""
[bold]Name:[/bold] {source.name}
[bold]ID:[/bold] {source.id}
[bold]Type:[/bold] {source.source_type.value}
[bold]Active:[/bold] {"Yes" if source.is_active else "No"}
[bold]Documents:[/bold] {len(docs)}
[bold]Created:[/bold] {source.created_at.strftime("%Y-%m-%d %H:%M")}
[bold]Last Synced:[/bold] {source.last_synced.strftime("%Y-%m-%d %H:%M") if source.last_synced else "Never"}
"""

    console.print(Panel(panel_content, title=f"Source: {source.name}"))


# ==================== Sync Commands ====================


@sync_app.command("run")
def sync_run(
    source_id: Optional[str] = typer.Argument(None, help="Source ID to sync (all if not specified)"),
):
    """Run synchronization for sources."""
    scheduler = get_scheduler()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing...", total=None)

        def on_complete(job):
            progress.update(task, completed=True)

        job = scheduler.trigger_sync(source_id=source_id, callback=on_complete)

    # 결과 표시 및 종료 코드 결정
    exit_code = ExitCode.SUCCESS
    if job.status.value == "completed":
        console.print("[green]✓ Sync completed successfully![/green]")
    elif job.status.value == "partial":
        console.print("[yellow]⚠ Sync completed with errors.[/yellow]")
        exit_code = ExitCode.SYNC_ERROR
    else:
        console.print(f"[red]✗ Sync failed: {job.error_message}[/red]")
        exit_code = ExitCode.SYNC_ERROR

    # 통계 표시
    table = Table(title="Sync Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Documents Processed", str(job.documents_processed))
    table.add_row("Documents Created", str(job.documents_created))
    table.add_row("Documents Updated", str(job.documents_updated))
    table.add_row("Documents Deleted", str(job.documents_deleted))
    table.add_row("Chunks Created", str(job.chunks_created))
    table.add_row("Errors", str(len(job.errors)))
    if job.duration_seconds:
        table.add_row("Duration", f"{job.duration_seconds:.2f}s")

    console.print(table)

    # 에러 표시
    if job.errors:
        console.print("\n[red]Errors:[/red]")
        for error in job.errors[:5]:  # 처음 5개만 표시
            console.print(f"  • {error.error_type}: {error.message}")
        if len(job.errors) > 5:
            console.print(f"  ... and {len(job.errors) - 5} more")

    # 종료 코드 반환
    if exit_code != ExitCode.SUCCESS:
        raise typer.Exit(exit_code)


@sync_app.command("history")
def sync_history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of jobs to show"),
):
    """Show sync job history."""
    storage = get_storage()
    jobs = storage.get_sync_history(limit=limit)

    if not jobs:
        console.print("[yellow]No sync history.[/yellow]")
        return

    table = Table(title="Sync History")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Trigger", style="blue")
    table.add_column("Processed", style="green")
    table.add_column("Errors", style="red")
    table.add_column("Started", style="yellow")
    table.add_column("Duration", style="white")

    for job in jobs:
        status_color = {
            "completed": "green",
            "partial": "yellow",
            "failed": "red",
            "running": "blue",
            "pending": "white",
        }.get(job.status.value, "white")

        started = job.started_at.strftime("%Y-%m-%d %H:%M") if job.started_at else "-"
        duration = f"{job.duration_seconds:.1f}s" if job.duration_seconds else "-"

        table.add_row(
            job.id[:8],
            f"[{status_color}]{job.status.value}[/{status_color}]",
            job.trigger.value,
            str(job.documents_processed),
            str(len(job.errors)),
            started,
            duration,
        )

    console.print(table)


# ==================== Search Command ====================


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
    source_id: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source ID"),
    source_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by source type"),
):
    """Search indexed documents."""
    indexer = get_indexer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Searching...", total=None)

        results = indexer.search(
            query=query,
            limit=limit,
            source_id=source_id,
            source_type=source_type,
        )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(results)} results for:[/bold] {query}\n")

    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score", 0)

        # Truncate text for display
        text = payload.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."

        panel_content = f"""
[bold]Score:[/bold] {score:.4f}
[bold]Source:[/bold] {payload.get("source_type", "unknown")}
[bold]Document:[/bold] {payload.get("title", "Untitled")}
[bold]URL:[/bold] {payload.get("url", "-")}

{text}
"""
        console.print(Panel(panel_content, title=f"Result {i}"))


# ==================== Status Command ====================


@app.command("status")
def status():
    """Show system status."""
    storage = get_storage()
    indexer = get_indexer()
    scheduler = get_scheduler()

    # Get counts
    sources = storage.get_sources()
    docs = storage.get_documents()
    last_job = storage.get_last_sync_job()

    try:
        stats = indexer.get_collection_stats()
        vector_count = stats.get("total_chunks", 0)
    except Exception:
        vector_count = "N/A"

    # Build status panel
    panel_content = f"""
[bold cyan]Sources[/bold cyan]
  Total: {len(sources)}
  Active: {sum(1 for s in sources if s.is_active)}

[bold cyan]Documents[/bold cyan]
  Total: {len(docs)}
  Indexed: {sum(1 for d in docs if d.indexed_at)}

[bold cyan]Vector Store[/bold cyan]
  Collection: {indexer.collection_name}
  Chunks: {vector_count}

[bold cyan]Scheduler[/bold cyan]
  Running: {"Yes" if scheduler.is_running() else "No"}
  Next Run: {scheduler.get_next_run().strftime("%Y-%m-%d %H:%M") if scheduler.get_next_run() else "Not scheduled"}

[bold cyan]Last Sync[/bold cyan]
  Status: {last_job.status.value if last_job else "No syncs yet"}
  Time: {last_job.started_at.strftime("%Y-%m-%d %H:%M") if last_job and last_job.started_at else "-"}
"""

    console.print(Panel(panel_content, title="RAG Indexer Status"))


# ==================== Scheduler Commands ====================


@scheduler_app.command("start")
def scheduler_start(
    cron: Optional[str] = typer.Option(None, "--cron", "-c", help="Cron expression"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
):
    """Start the scheduler."""
    scheduler = get_scheduler(cron_expression=cron)

    scheduler.start()
    console.print(f"[green]✓ Scheduler started.[/green]")
    console.print(f"  Cron: {scheduler.cron_expression}")

    if foreground:
        console.print("[yellow]Running in foreground. Press Ctrl+C to stop.[/yellow]")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            console.print("\n[yellow]Scheduler stopped.[/yellow]")


@scheduler_app.command("stop")
def scheduler_stop():
    """Stop the scheduler."""
    scheduler = get_scheduler()

    if not scheduler.is_running():
        console.print("[yellow]Scheduler is not running.[/yellow]")
        return

    scheduler.stop()
    console.print("[green]✓ Scheduler stopped.[/green]")


@scheduler_app.command("status")
def scheduler_status():
    """Show scheduler status."""
    scheduler = get_scheduler()

    if scheduler.is_running():
        next_run = scheduler.get_next_run()
        console.print("[green]Scheduler is running.[/green]")
        console.print(f"  Cron: {scheduler.cron_expression}")
        console.print(f"  Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else 'N/A'}")
    else:
        console.print("[yellow]Scheduler is not running.[/yellow]")


# ==================== Main Entry Point ====================


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Output logs as JSON"),
):
    """RAG Document Indexer CLI."""
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level, json_format=json_logs)


def cli():
    """CLI 진입점.

    전역 예외 처리를 통해 적절한 종료 코드 반환.
    """
    try:
        app()
    except ConnectionError as e:
        # 연결 오류 (Qdrant, Notion API, 네트워크)
        console.print(f"[red]Connection error: {e}[/red]")
        raise SystemExit(ExitCode.CONNECTION_ERROR)
    except KeyboardInterrupt:
        # 사용자 중단
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise SystemExit(ExitCode.SUCCESS)
    except Exception as e:
        # 예상치 못한 내부 오류
        console.print(f"[red]Internal error: {e}[/red]")
        raise SystemExit(ExitCode.INTERNAL_ERROR)


if __name__ == "__main__":
    cli()
