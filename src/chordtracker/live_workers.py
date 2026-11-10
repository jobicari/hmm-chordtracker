"""Live sessions in worker processes: the demo server's decoding, off its event loop and its interpreter lock.

The demo server (tornado, one process) only moves bytes. A fixed pool of worker processes does the
decoding: each loads the chain once, then runs the sessions pinned to it, one message at a time in the
order they came, so a session's blocks and labels keep their order. A step holds Python's interpreter
lock for much of its time, so sessions in threads of one process share about one CPU; in processes they
use one CPU each.

    workers = Workers(factory, n)   n processes (spawn), each calling factory() once: a picklable callable
                                    returning a chain with info() -> dict (sent once, as workers.info) and
                                    new_session() -> an object with labels ({point: [class index]}),
                                    step(float32 block) and finish(), as play_along.Analysis
    await workers.start()           spawn them on the running loop; returns once every one has loaded
    workers.open(sid, on_message)   pin session sid to the least-loaded worker that is up (False: none is)
    workers.audio(sid, pcm)         16-bit mono PCM bytes -> on_message("labels", new) once it is decoded
    workers.end(sid)                -> on_message("labels", the rest), then on_message("done", None)
    workers.close(sid)              forget the session (the connection has closed)
    workers.alive                   workers loaded and running
    await workers.stop()            stop them (the server is shutting down)

new: {point: {"from": k, "labels": [...]}} for the points with new labels ({} if none). A failing session
gets on_message("error", message). A worker that dies ends its sessions with an error and is restarted.
"""
import asyncio
import multiprocessing
import signal
import sys

import numpy as np

RESTART_DELAY_S = 1.0


def _new_labels(analysis, sent: dict) -> dict:
    """The labels of each point that have not been sent yet, and mark them sent."""
    new = {}
    for point, labels in analysis.labels.items():
        k = sent.get(point, 0)
        if len(labels) > k:
            new[point] = {"from": k, "labels": [int(x) for x in labels[k:]]}
            sent[point] = len(labels)
    return new


def _worker(factory, inbox, conn) -> None:
    """A worker process: load the chain, then serve its sessions' messages in order until "stop"."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)               # the server stops its workers itself
    chain = factory()
    conn.send(("hello", chain.info()))
    sessions = {}                                               # sid -> [analysis, sent]
    while True:
        message = inbox.get()
        kind = message[0]
        if kind == "stop":
            return
        sid = message[1]
        try:
            if kind == "start":
                sessions[sid] = [chain.new_session(), {}]
            elif kind == "close":
                sessions.pop(sid, None)
            elif sid in sessions:
                analysis, sent = sessions[sid]
                if kind == "audio":
                    analysis.step(np.frombuffer(message[2], dtype="<i2").astype(np.float32) / 32768.0)
                    conn.send((sid, "labels", _new_labels(analysis, sent)))
                elif kind == "end":
                    analysis.finish()
                    conn.send((sid, "labels", _new_labels(analysis, sent)))
                    conn.send((sid, "done", None))
                    del sessions[sid]
        except Exception as e:                                  # one session failed: end it, keep serving
            sessions.pop(sid, None)
            conn.send((sid, "error", f"Decoding failed ({type(e).__name__}: {e})."))


class _Slot:
    """One worker process, its inbox, the read end of its pipe, and the sessions pinned to it."""

    def __init__(self, process, inbox, conn):
        self.process, self.inbox, self.conn = process, inbox, conn
        self.sessions, self.ready = set(), False


class Workers:
    def __init__(self, factory, n: int):
        if n < 1:
            raise ValueError("at least one worker")
        self.factory, self.n = factory, n
        self.context = multiprocessing.get_context("spawn")
        self.slots, self.handlers, self.info = [], {}, None
        self.loop, self.changed, self.stopping = None, None, False

    async def start(self, timeout: float = 300.0) -> None:
        self.loop, self.changed = asyncio.get_running_loop(), asyncio.Event()
        self.slots = [self._spawn() for _ in range(self.n)]
        async with asyncio.timeout(timeout):
            while self.alive < self.n:
                if not any(s.process.is_alive() or s.ready for s in self.slots):
                    raise RuntimeError("the decoding workers could not start")
                self.changed.clear()
                await self.changed.wait()

    @property
    def alive(self) -> int:
        return sum(s.ready and s.process.is_alive() for s in self.slots)

    def open(self, sid, on_message) -> bool:
        ready = [s for s in self.slots if s.ready and s.process.is_alive()]
        if not ready:
            return False
        slot = min(ready, key=lambda s: len(s.sessions))
        slot.sessions.add(sid)
        self.handlers[sid] = (slot, on_message)
        slot.inbox.put(("start", sid))
        return True

    def audio(self, sid, pcm: bytes) -> None:
        self._send(sid, ("audio", sid, pcm))

    def end(self, sid) -> None:
        self._send(sid, ("end", sid))

    def close(self, sid) -> None:
        handler = self.handlers.pop(sid, None)
        if handler is not None:
            handler[0].sessions.discard(sid)
            handler[0].inbox.put(("close", sid))

    async def stop(self, timeout: float = 5.0) -> None:
        self.stopping = True
        for slot in self.slots:
            self.loop.remove_reader(slot.conn.fileno())
            slot.inbox.put(("stop",))
        for slot in self.slots:
            await asyncio.to_thread(slot.process.join, timeout)
            if slot.process.is_alive():
                slot.process.terminate()
            slot.conn.close()
            slot.inbox.cancel_join_thread()
            slot.inbox.close()

    # ---------------------------------------------------------------- internals
    def _send(self, sid, message) -> None:
        handler = self.handlers.get(sid)
        if handler is not None:
            handler[0].inbox.put(message)

    def _spawn(self) -> _Slot:
        inbox = self.context.Queue()
        read, write = self.context.Pipe(duplex=False)
        process = self.context.Process(target=_worker, args=(self.factory, inbox, write), daemon=True)
        process.start()
        write.close()                                           # the read end then sees EOF when the worker dies
        slot = _Slot(process, inbox, read)
        self.loop.add_reader(read.fileno(), self._readable, slot)
        return slot

    def _readable(self, slot: _Slot) -> None:
        try:
            while slot.conn.poll():
                message = slot.conn.recv()
                if message[0] == "hello":
                    self.info, slot.ready = message[1], True
                    self.changed.set()
                    continue
                sid, kind, payload = message
                handler = self.handlers.get(sid)
                if handler is None or handler[0] is not slot:
                    continue                                    # the connection has closed meanwhile
                if kind in ("done", "error"):
                    self.handlers.pop(sid)
                    slot.sessions.discard(sid)
                handler[1](kind, payload)
        except (EOFError, OSError):
            self._died(slot)

    def _died(self, slot: _Slot) -> None:
        self.loop.remove_reader(slot.conn.fileno())
        slot.conn.close()
        slot.inbox.cancel_join_thread()
        slot.inbox.close()
        slot.process.join(timeout=1.0)
        slot.ready = False
        self.changed.set()
        for sid in list(slot.sessions):
            _, on_message = self.handlers.pop(sid)
            on_message("error", "The decoder stopped; start again.")
        slot.sessions.clear()
        if not self.stopping:
            print(f"decoding worker {slot.process.pid} stopped (exit code {slot.process.exitcode}); restarting",
                  file=sys.stderr, flush=True)
            self.loop.call_later(RESTART_DELAY_S, self._restart, slot)

    def _restart(self, slot: _Slot) -> None:
        if not self.stopping and slot in self.slots:
            self.slots[self.slots.index(slot)] = self._spawn()
