"""Keep optional native game engines and C bot globals in one episode process."""
from __future__ import annotations

import multiprocessing as mp


def _worker(connection, game, horizon):
    adapter = None
    try:
        while True:
            method, kwargs = connection.recv()
            if method == "close":
                break
            try:
                if adapter is None:
                    if game == "rps":
                        from .rps import RoshamboAdapter
                        adapter = RoshamboAdapter(horizon=horizon)
                    elif game == "fixed_rps":
                        from .fixed_rps import FixedRPSAdapter
                        adapter = FixedRPSAdapter(horizon=horizon)
                    elif game == "hanabi":
                        from .hanabi import HanabiAdapter, RuleHanabiPartner
                        adapter = HanabiAdapter(RuleHanabiPartner(kwargs["opponent_id"]))
                    else:
                        raise ValueError("unknown isolated game")
                result = getattr(adapter, method)(**kwargs)
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, (type(exc).__name__, str(exc))))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if adapter is not None:
            adapter.close()
        connection.close()


class IsolatedGameAdapter:
    def __init__(self, game, *, horizon, timeout=30):
        self.game, self.horizon, self.timeout = game, horizon, timeout
        self.game_id = {"rps": "open_spiel_roshambo", "fixed_rps": "open_spiel_roshambo",
                        "hanabi": "adaptive_hanabi"}[game]
        self._process = self._connection = None

    def _call(self, method, **kwargs):
        if self._process is None:
            raise RuntimeError("reset must be called before using the adapter")
        try:
            self._connection.send((method, kwargs))
            if not self._connection.poll(self.timeout):
                raise RuntimeError("Game engine timed out")
            success, value = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("Game engine process exited; check its optional dependencies") from exc
        if not success:
            raise RuntimeError(f"Game engine {value[0]}: {value[1]}")
        return value

    def reset(self, **kwargs):
        self.close()
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, self.game, self.horizon), daemon=True)
        self._process.start()
        child.close()
        return self._call("reset", **kwargs)

    def step(self, action):
        return self._call("step", action=action)

    def fingerprint(self):
        return self._call("fingerprint")

    def close(self):
        if self._process is not None:
            try:
                if self._process.is_alive():
                    try:
                        self._connection.send(("close", {}))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    self._process.join(2)
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(2)
            finally:
                self._connection.close()
                if not self._process.is_alive():
                    self._process.close()
                self._process = self._connection = None
