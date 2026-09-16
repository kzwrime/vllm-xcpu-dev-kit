"""Exercise real shell cleanup with failures, signals and stale run logs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('run_code,cleanup_code,mode,category,expected', [
    (0, 0, 'test', 'success', 0),
    (0, 7, 'test', 'failed', 7),
    (9, 7, 'test', 'failed', 9),
    (130, 0, 'test', 'failed', 130),
    (143, 0, 'test', 'failed', 143),
    (0, 0, 'none', 'stopped', 0),
])
def test_archive_once_even_on_cleanup_failure(tmp_path, run_code, cleanup_code, mode, category, expected):
    stale = tmp_path / 'previous.log'
    stale.write_text('ERROR previous run\n')
    script = r'''
set -eo pipefail
source "$REPO/vllm_scripts/launcher_common.sh"
source "$REPO/vllm_scripts/test_log_lifecycle.sh"
LOG_ROOT="$TEST_ROOT"
SUCCESS_ROOT="$LOG_ROOT/success"
FAILED_ROOT="$LOG_ROOT/failed"
RUN_START_TS=20260915
PRESET_TAG=regression
TEST_MODE="$MODE"
launcher_cleanup_processes() { echo cleanup >> "$TEST_ROOT/cleanup_calls"; return "$CLEANUP_CODE"; }
test_prepare_log_directory
LAUNCH_LOG="$LOG_DIR/launch.log"
echo 'current output' > "$LAUNCH_LOG"
test -z "$(launcher_collect_error_details)"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "$RUN_CODE" in
    130) kill -INT $$ ;;
    143) kill -TERM $$ ;;
    *) exit "$RUN_CODE" ;;
esac
'''
    env = {**os.environ, 'REPO': str(ROOT), 'TEST_ROOT': str(tmp_path),
           'RUN_CODE': str(run_code), 'CLEANUP_CODE': str(cleanup_code), 'MODE': mode}
    for attempt in range(2):
        result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        archives = list((tmp_path / category).iterdir())
        assert len(archives) == attempt + 1
        assert not list((tmp_path / 'runs').iterdir())
        for archive in archives:
            assert json.loads((archive / 'run_result.json').read_text()) == {
                'run_exit_code': run_code, 'cleanup_exit_code': cleanup_code, 'exit_code': expected}
            assert (archive / 'launch.log').read_text() == 'current output\n'
    assert (tmp_path / 'cleanup_calls').read_text() == 'cleanup\ncleanup\n'
    assert stale.read_text() == 'ERROR previous run\n'


def test_archive_failure_preserves_original_directory(tmp_path):
    script = r'''
set -eo pipefail
source "$REPO/vllm_scripts/launcher_common.sh"
source "$REPO/vllm_scripts/test_log_lifecycle.sh"
LOG_ROOT="$TEST_ROOT"; SUCCESS_ROOT="$LOG_ROOT/success"; FAILED_ROOT="$LOG_ROOT/failed"
RUN_START_TS=now; PRESET_TAG=test; TEST_MODE=test
test_prepare_log_directory
launcher_cleanup_processes() { return 0; }
mv() { return 8; }
trap cleanup EXIT
exit 0
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True,
                            env={**os.environ, 'REPO': str(ROOT), 'TEST_ROOT': str(tmp_path)})
    assert result.returncode == 8
    records = list((tmp_path / 'runs').glob('*/run_result.json'))
    assert len(records) == 1
    assert json.loads(records[0].read_text())['exit_code'] == 8
    assert json.loads(records[0].read_text())['archive_exit_code'] == 8
    assert not list((tmp_path / 'success').iterdir())


@pytest.mark.parametrize('options,expected', [
    ('--revision release --trust-remote-code --all2all-backend mpi_alltoallv_v7', ['--revision','release','--trust-remote-code']),
    ('--revision=abc --no-trust-remote-code', ['--revision=abc','--no-trust-remote-code']),
    ('--revision abc\n--trust-remote-code', ['--revision','abc','--trust-remote-code']),
    ('', []),
])
def test_f_model_load_passthrough(options, expected):
    script = 'source "$REPO/vllm_scripts/serve/af_model_load_args.sh"; af_model_load_args; printf "%s\\n" "${AF_MODEL_LOAD_ARGS[@]}"'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True,
                            env={**os.environ, 'REPO': str(ROOT), 'VLLM_OPTIONAL_ARGS': options})
    assert result.stdout.splitlines() == (expected or [''])


@pytest.mark.parametrize('external_log_dir', [False, True])
def test_mp_template_preserves_failure_and_routes_log(tmp_path, external_log_dir):
    import shutil
    serve = tmp_path / 'serve'
    serve.mkdir()
    template = serve / 'serve_mp_template.sh'
    shutil.copyfile(ROOT/'vllm_scripts/serve/serve_mp_template.sh', template)
    (tmp_path/'common.sh').write_text('''
parse_args_and_load_env() { :; }
check_and_print_env() { :; }
vllm() { echo 'injected serve failure'; return 23; }
''')
    log_dir = tmp_path/('isolated run' if external_log_dir else 'logs')
    env = dict(os.environ)
    env.pop('VLLM_RUN_LOG_DIR', None)
    if external_log_dir:
        env['VLLM_RUN_LOG_DIR'] = str(log_dir)
    result = subprocess.run(['bash',str(template)],cwd=tmp_path,env=env,capture_output=True)
    assert result.returncode == 23
    assert (log_dir/'vllm_serve_log.txt').read_text() == 'injected serve failure\n'
