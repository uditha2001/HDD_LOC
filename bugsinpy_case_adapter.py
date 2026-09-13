"""Reusable Docker adapter for small structured BugsInPy scenarios.

Each supported case receives a JSON tree.  A case-specific handler interprets
that tree as the public/program input needed to exercise the bug.  The exact
same handler and candidate are run against buggy and fixed checkouts; only the
buggy execution is measured for coverage.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
BUGS_WORKSPACE_ROOT = REPO_ROOT.parent / "BugsInPy" / "bugs_workspace"
CONTAINER_WORKSPACE_ROOT = Path("/workspace")


@dataclass(frozen=True)
class CaseEvaluationResult:
    outcome: str
    buggy_output: dict
    fixed_output: dict
    coverage: set[tuple[str, int]]
    structurally_valid: bool
    semantically_valid: bool
    runtime_ms: float


class BugsInPyCaseAdapter:
    """Execute one named scenario against a BugsInPy checkout pair."""

    def __init__(
        self,
        case_name: str,
        buggy_root: Path,
        fixed_root: Path,
        coverage_source: str,
        *,
        timeout: int = 30,
        collect_coverage: bool = True,
    ) -> None:
        self.case_name = case_name
        self.buggy_root = Path(buggy_root)
        self.fixed_root = Path(fixed_root)
        self.coverage_source = coverage_source
        self.timeout = timeout
        self.collect_coverage = collect_coverage
        self.workspace_root = self._detect_workspace_root(self.buggy_root, self.fixed_root)
        self._validate_paths()

    def evaluate(self, candidate: Any) -> CaseEvaluationResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="hddloc_case_", dir=str(self.workspace_root)) as temp:
            temp_dir = Path(temp)
            candidate_file = temp_dir / "candidate.json"
            candidate_file.write_text(
                json.dumps(candidate, sort_keys=True, ensure_ascii=False), encoding="utf-8"
            )
            runner_file = temp_dir / "runner.py"
            runner_file.write_text(self._runner_source(), encoding="utf-8")
            coverage_file = temp_dir / ".coverage"

            buggy = self._run_version(
                self.buggy_root, candidate_file, runner_file, coverage_file, self.collect_coverage
            )
            fixed = self._run_version(
                self.fixed_root, candidate_file, runner_file, None, False
            )
            outcome, structurally_valid, semantically_valid = self._classify(buggy, fixed)
            coverage = set()
            if self.collect_coverage and coverage_file.exists():
                coverage = self._read_coverage(coverage_file, temp_dir)
            return CaseEvaluationResult(
                outcome=outcome,
                buggy_output=buggy,
                fixed_output=fixed,
                coverage=coverage,
                structurally_valid=structurally_valid,
                semantically_valid=semantically_valid,
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

    @staticmethod
    def _classify(buggy: dict, fixed: dict) -> tuple[str, bool, bool]:
        if buggy.get("kind") == "value" and fixed.get("kind") == "value":
            return (
                ("FAIL", True, True)
                if buggy.get("output") != fixed.get("output")
                else ("PASS", True, True)
            )
        if buggy.get("kind") == "exception" and fixed.get("kind") == "exception":
            if (
                buggy.get("exception_type") == fixed.get("exception_type")
                and buggy.get("message") == fixed.get("message")
            ):
                return "INVALID", True, False
            return "FAIL", True, True
        return "FAIL", True, True

    def _run_version(
        self,
        project_root: Path,
        candidate_file: Path,
        runner_file: Path,
        coverage_file: Path | None,
        use_coverage: bool,
    ) -> dict:
        project = self._container_path(project_root)
        candidate = self._container_path(candidate_file)
        runner = self._container_path(runner_file)
        work_dir = self._container_path(candidate_file.parent)
        arguments = (
            f"{shlex.quote(str(runner))} {shlex.quote(str(project))} "
            f"{shlex.quote(str(candidate))} {shlex.quote(self.case_name)}"
        )
        if use_coverage:
            if coverage_file is None:
                raise ValueError("coverage_file is required when coverage is enabled")
            coverage = self._container_path(coverage_file)
            python_command = (
                f"export COVERAGE_FILE={shlex.quote(str(coverage))}; "
                f"python3 -m coverage run --source={shlex.quote(self.coverage_source)} {arguments}"
            )
        else:
            python_command = f"python3 {arguments}"
        command = [
            "docker",
            "exec",
            "bugsinpy_bg",
            "bash",
            "-c",
            (
                "export PATH=$PATH:/BugsInPy/framework/bin:/framework/bin; "
                "export LC_ALL=C.UTF-8 LANG=C.UTF-8; "
                f"cd {shlex.quote(str(work_dir))} && {python_command}"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            return {"kind": "exception", "exception_type": type(exc).__name__, "message": str(exc)}
        except subprocess.TimeoutExpired:
            return {"kind": "exception", "exception_type": "TimeoutExpired", "message": "Execution timed out"}
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                "kind": "exception",
                "exception_type": "RunnerFailure",
                "message": (
                    f"returncode={result.returncode}; stdout={result.stdout!r}; "
                    f"stderr={result.stderr!r}"
                ),
            }

    def _read_coverage(self, coverage_file: Path, temp_dir: Path) -> set[tuple[str, int]]:
        output_file = temp_dir / "coverage.json"
        command = [
            "docker",
            "exec",
            "bugsinpy_bg",
            "bash",
            "-c",
            (
                f"export COVERAGE_FILE={shlex.quote(str(self._container_path(coverage_file)))}; "
                f"python3 -m coverage json -o {shlex.quote(str(self._container_path(output_file)))}"
            ),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not output_file.exists():
            return set()
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()

        executed: set[tuple[str, int]] = set()
        project_root = self.buggy_root.resolve()
        for filename, information in data.get("files", {}).items():
            path = Path(filename)
            try:
                host_path = self.workspace_root / path.relative_to(CONTAINER_WORKSPACE_ROOT)
            except ValueError:
                host_path = path
            try:
                relative = host_path.relative_to(project_root)
            except ValueError:
                continue
            if "test" in relative.parts or "tests" in relative.parts:
                continue
            for line in information.get("executed_lines", []):
                executed.add((str(relative), int(line)))
        return executed

    @staticmethod
    def _runner_source() -> str:
        # Keep this source compatible with the Python 3.6 image used by the
        # BugsInPy container.
        return r'''
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
candidate_file = Path(sys.argv[2]).resolve()
case_name = sys.argv[3]
sys.path.insert(0, str(project_root))
candidate = json.loads(candidate_file.read_text(encoding='utf-8'))


def run_black_1(c):
    sys.modules['_black_version'] = types.SimpleNamespace(version='hdd-loc')
    import black

    class UnavailableExecutor(object):
        def __init__(self, *args, **kwargs):
            raise OSError('multiprocessing unavailable')

    async def no_formatting(**kwargs):
        return None

    multiprocessing_available = c.get('environment', {}).get('multiprocessing', True)
    if not multiprocessing_available:
        black.ProcessPoolExecutor = UnavailableExecutor
    black.schedule_formatting = no_formatting
    black.shutdown = lambda loop: loop.close()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sources = set(Path(name) for name in c.get('files', []))
    black.reformat_many(sources, True, black.WriteBack.YES, black.FileMode(), black.Report())
    return {'completed': True, 'source_count': len(sources)}


def run_black_3(c):
    sys.modules['_black_version'] = types.SimpleNamespace(version='hdd-loc')
    import black
    from click.testing import CliRunner

    source = candidate_file.parent / 'black3_input.py'
    source.write_text(c.get('source', 'x = 1\n'), encoding='utf-8')
    arguments = c.get('arguments', {})
    if arguments.get('use_config', False):
        missing_config = candidate_file.parent / c.get('missing_config_name', 'missing.toml')
        if missing_config.exists():
            missing_config.unlink()
        cli_arguments = ['--config', str(missing_config), str(source)]
    else:
        cli_arguments = [str(source)]
    result = CliRunner().invoke(black.main, cli_arguments)
    return {
        'exit_code': result.exit_code,
        'output': result.output,
        'exception_type': type(result.exception).__name__ if result.exception else None,
        'exception_message': str(result.exception) if result.exception else None,
    }


def run_tornado_1(c):
    from tornado.websocket import WebSocketHandler

    calls = []
    class Connection(object):
        def set_nodelay(self, value):
            calls.append(bool(value))
    class Handler(object):
        ws_connection = Connection()
        stream = Connection() if not c.get('connection', {}).get('established', False) else None
    WebSocketHandler.set_nodelay(Handler(), c.get('value', False))
    return {'calls': calls}


def run_tornado_2(c):
    from tornado import httputil
    from tornado.concurrent import Future
    from tornado.http1connection import HTTP1Connection

    class ClosedStream(object):
        def closed(self):
            return True
    connection = object.__new__(HTTP1Connection)
    connection.is_client = True
    connection.stream = ClosedStream()
    connection._write_future = None
    connection._pending_write = None
    start = httputil.RequestStartLine(c.get('method', 'GET'), c.get('path', '/'), 'HTTP/1.1')
    headers = httputil.HTTPHeaders(c.get('headers', {}))
    HTTP1Connection.write_headers(connection, start, headers)
    return {'chunking_output': connection._chunking_output,
            'headers': sorted(headers.get_all())}


def run_tornado_3(c):
    from tornado.httpclient import AsyncHTTPClient

    client = object.__new__(AsyncHTTPClient)
    client._closed = False
    client.io_loop = object()
    cache_state = c.get('cache_state', 'self')
    if cache_state == 'self':
        client._instance_cache = {client.io_loop: client}
    elif c['cache_state'] == 'other':
        client._instance_cache = {client.io_loop: object()}
    else:
        client._instance_cache = {}
    AsyncHTTPClient.close(client)
    return {'closed': client._closed, 'cache_size': len(client._instance_cache)}


def run_tornado_4(c):
    from tornado.web import StaticFileHandler

    statuses = []
    response_headers = {}
    range_spec = c.get('range', {'kind': 'explicit', 'start': 0, 'end': 3})
    if range_spec['kind'] == 'suffix':
        range_value = 'bytes=-%s' % range_spec['length']
    else:
        range_value = 'bytes=%s-%s' % (range_spec['start'], range_spec['end'])

    class Request(object):
        headers = {'Range': range_value}
    class Handler(object):
        request = Request()
        root = '/unused'
        def parse_url_path(self, path): return path
        def get_absolute_path(self, root, path): return path
        def validate_absolute_path(self, root, path): return path
        def get_modified_time(self): return None
        def set_headers(self): return None
        def should_return_304(self): return False
        def get_content_size(self): return c.get('resource_size', 26)
        def set_status(self, status): statuses.append(status)
        def set_header(self, name, value): response_headers[name] = value
        def get_content(self, path, start=None, end=None): return b''
        def write(self, chunk): return None
        async def flush(self): return None
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(StaticFileHandler.get(Handler(), 'resource', True))
    finally:
        loop.close()
    return {'statuses': statuses, 'headers': response_headers}


def run_youtube_dl_1(c):
    from youtube_dl.utils import match_str

    expressions = []
    for clause in c.get('clauses', [{'key': 'title'}]):
        if 'value' in clause:
            expression = clause['key'] + clause.get('op', '') + str(clause['value'])
        else:
            expression = clause.get('op', '') + clause['key']
        expressions.append(expression)
    if not expressions:
        raise ValueError('at least one filter clause is required')
    return {'matches': match_str(' & '.join(expressions), c.get('metadata', {'title': 'abc'}))}


def run_tqdm_1(c):
    from tqdm.contrib import tenumerate

    def passthrough(iterable, *args, **kwargs):
        return iterable
    return {'items': list(tenumerate(c.get('iterable', []), c.get('start', 0), tqdm_class=passthrough))}


def run_tqdm_2(c):
    from tqdm import tqdm

    options = dict(c.get('options', {}))
    return {'formatted': tqdm.format_meter(
        c.get('n', 0), c.get('total', 100), c.get('elapsed', 1), **options)}


def run_luigi_1(c):
    import importlib.util

    class Config(object):
        pass
    class Parameter(object):
        def __init__(self, *args, **kwargs):
            pass
    luigi_stub = types.ModuleType('luigi')
    luigi_stub.__path__ = [str(project_root / 'luigi')]
    luigi_stub.Config = Config
    luigi_stub.parameter = types.SimpleNamespace(
        BoolParameter=Parameter,
        ListParameter=Parameter,
        IntParameter=Parameter,
        Parameter=Parameter,
    )
    scheduler_stub = types.ModuleType('luigi.scheduler')
    scheduler_stub.Scheduler = object
    scheduler_stub.RPC_METHODS = set()
    sys.modules['luigi'] = luigi_stub
    sys.modules['luigi.scheduler'] = scheduler_stub

    tornado_stub = types.ModuleType('tornado')
    tornado_stub.__path__ = []
    for module_name in ('httpserver', 'ioloop', 'netutil', 'web'):
        module = types.ModuleType('tornado.' + module_name)
        setattr(tornado_stub, module_name, module)
        sys.modules['tornado.' + module_name] = module
    tornado_stub.web.RequestHandler = object
    tornado_stub.web.Application = object
    tornado_stub.httpserver.HTTPServer = object
    tornado_stub.ioloop.IOLoop = object
    tornado_stub.ioloop.PeriodicCallback = object
    sys.modules['tornado'] = tornado_stub

    spec = importlib.util.spec_from_file_location(
        'luigi.server', str(project_root / 'luigi' / 'server.py'))
    server = importlib.util.module_from_spec(spec)
    sys.modules['luigi.server'] = server
    spec.loader.exec_module(server)
    MetricsHandler = server.MetricsHandler

    calls = []
    class Payload(object):
        def configure_http_handler(self, handler): calls.append('payload')
        def __bool__(self): return c.get('metrics_available', False)
        __nonzero__ = __bool__
    class Collector(object):
        def generate_latest(self): return Payload()
        def configure_http_handler(self, handler): calls.append('collector')
    class State(object): _metrics_collector = Collector()
    class Scheduler(object): _state = State()
    class Handler(object):
        _scheduler = Scheduler()
        def write(self, metrics): calls.append('write')
    MetricsHandler.get(Handler())
    return {'calls': calls}


handlers = {
    'black_1': run_black_1,
    'black_3': run_black_3,
    'tornado_1': run_tornado_1,
    'tornado_2': run_tornado_2,
    'tornado_3': run_tornado_3,
    'tornado_4': run_tornado_4,
    'youtube-dl_1': run_youtube_dl_1,
    'tqdm_1': run_tqdm_1,
    'tqdm_2': run_tqdm_2,
    'luigi_1': run_luigi_1,
}

try:
    output = handlers[case_name](candidate)
    print(json.dumps({'kind': 'value', 'output': output}, sort_keys=True))
except Exception as exc:
    print(json.dumps({'kind': 'exception',
                      'exception_type': type(exc).__name__,
                      'message': str(exc)}, sort_keys=True))
'''

    def _validate_paths(self) -> None:
        if not self.buggy_root.exists():
            raise FileNotFoundError(f"buggy checkout missing: {self.buggy_root}")
        if not self.fixed_root.exists():
            raise FileNotFoundError(f"fixed checkout missing: {self.fixed_root}")

    @staticmethod
    def _detect_workspace_root(buggy_root: Path, fixed_root: Path) -> Path:
        for root in (buggy_root, fixed_root):
            resolved = root.resolve()
            for ancestor in (resolved, *resolved.parents):
                if ancestor.name == "bugs_workspace":
                    return ancestor
        return Path(os.path.commonpath([str(buggy_root.resolve()), str(fixed_root.resolve())]))

    def _container_path(self, host_path: Path) -> Path:
        relative = Path(host_path).resolve().relative_to(self.workspace_root.resolve())
        return CONTAINER_WORKSPACE_ROOT / relative
