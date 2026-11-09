"""Run-local, scope-specific model replies; successful calls survive a restart."""
from pathlib import Path

from oppomem.runtime.storage import digest, read, write


class CachedLLM:
    def __init__(self, client, directory, scope="run", *, read_only_cache=None):
        self.client, self.directory, self.scope = client, Path(directory), scope
        self.read_only_cache = Path(read_only_cache) if read_only_cache else None
        if self.read_only_cache is not None and self.read_only_cache.resolve() == self.directory.resolve():
            raise ValueError("Read-only model cache must differ from the writable cache")

    def scoped(self, scope):
        return CachedLLM(self.client, self.directory, scope, read_only_cache=self.read_only_cache)

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _call(self, method, kwargs):
        request = {"scope": self.scope, "method": method, "model": self.client.model,
                   "effort": self.client.reasoning_effort, "kwargs": kwargs}
        path = self.directory / (digest(request) + ".json")
        candidates = [path]
        if self.read_only_cache is not None:
            candidates.append(self.read_only_cache / path.name)
        for candidate in candidates:
            if not candidate.exists():
                continue
            saved = read(candidate)
            if saved["request"] != request or saved["sha256"] != digest(saved["response"]):
                raise ValueError("Model reply cache changed")
            if candidate != path:
                write(path, {"request": request, "response": saved["response"],
                             "sha256": saved["sha256"], "origin": str(candidate.resolve())})
            return saved["response"]
        response = getattr(self.client, method)(**kwargs)
        write(path, {"request": request, "response": response, "sha256": digest(response)})
        return response

    def complete_json(self, **kwargs):
        return self._call("complete_json", kwargs)

    def complete_text(self, **kwargs):
        return self._call("complete_text", kwargs)


def scoped(llm, scope):
    return llm.scoped(scope) if isinstance(llm, CachedLLM) else llm
