"""Durable evaluation lifecycle records, including startup failures."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class RunState:
    def __init__(self, output, configuration):
        self.output = Path(output)
        self.info = {'status': 'started', 'phase': 'initializing',
                     'configuration': configuration, 'processed_examples': 0,
                     'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
                     'started_at': datetime.now(timezone.utc).isoformat()}
        self.start = perf_counter()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        return self

    def save(self):
        self.info['elapsed_seconds'] = perf_counter() - self.start
        write_json(self.output / 'manifest.json', self.info)

    def phase(self, name):
        self.info['phase'] = name
        self.save()
        print(f'[{name}] processed={self.info["processed_examples"]}', flush=True)

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.info['status'] = 'interrupted' if issubclass(kind, KeyboardInterrupt) else 'failed'
            self.info['error_type'] = kind.__name__
        else:
            self.info['status'] = 'complete'
            self.info['phase'] = 'complete'
        self.info['finished_at'] = datetime.now(timezone.utc).isoformat()
        self.save()
        return False
