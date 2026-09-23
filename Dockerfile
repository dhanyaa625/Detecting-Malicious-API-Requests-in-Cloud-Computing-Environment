FROM python:3.11-slim

WORKDIR /app

# torch-geometric needs a C build toolchain for a couple of its optional
# extensions; keep the image slim but include what's needed to install cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only torch build first (~200MB vs ~2GB+ for the default
# CUDA wheel) -- this is inference-only on free-tier hosts with no GPU, and
# pip will see the already-satisfied torch>=2.0.0 requirement below and skip
# re-downloading it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# model.pt, data/, gnn_outputs.json etc. are copied in above. If you'd rather
# not bake the trained model into the image, mount a volume over /app/data
# and /app/model.pt instead -- see the deployment notes for why that matters
# once Agent 2 starts updating them at runtime.

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
