# The demo website with live decoding (scripts/demo_server.py): the page, the prepared samples and the
# model, on CPU. See demo/README.md, "Deploy with Docker".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-demo.txt /tmp/
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' /tmp/requirements-demo.txt)" \
 && pip install -r /tmp/requirements-demo.txt \
 && rm /tmp/requirements-demo.txt

RUN useradd --create-home --uid 10001 demo
WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY results/ results/
COPY demo/ demo/
# demo/samples is git-ignored (scripts/demo_export.py writes it): refuse to build a page without samples.
RUN test -f demo/samples/index.json || { echo "demo/samples/index.json missing: run scripts/demo_export.py first" >&2; exit 1; } \
 && chmod -R a+rX /app

USER demo
# One thread per library: the server runs one block at a time per session and sessions in parallel.
# Left at their defaults, OpenMP and OpenBLAS start a thread per CPU for every small product and run
# about 6 times slower in Docker Desktop's VM.
ENV PYTHONPATH=/app/src \
    NUMBA_CACHE_DIR=/home/demo/.cache/numba \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CHORD_DEMO_HOST=0.0.0.0
# Compile librosa's numba functions once, at build time (Chain warms itself up), so that every decoding
# worker finds them in the cache and starts quickly.
RUN python -c "import sys; sys.path.insert(0, 'scripts'); import demo_server; demo_server.Chain()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('CHORD_DEMO_PORT', os.environ.get('PORT', '8000')), timeout=4)"
CMD ["python", "scripts/demo_server.py"]
