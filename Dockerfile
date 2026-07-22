# Optional container image for fully reproducible analyses.
#
# Build:  docker build -t enhanced-rsm .
# Run:    docker run --rm enhanced-rsm el1 --h 449.8 --ho 337.35 --bf 152.4 \
#                 --tw 7.6 --tf 10.9 --r 10.2 --fy 355 --mv 1.333
#
# To keep output files, mount a directory and write into it:
#   docker run --rm -v "$PWD/out:/out" enhanced-rsm el1 ... --output /out/el1.json

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY enhanced_rsm/ ./enhanced_rsm/
RUN pip install --no-cache-dir .

ENTRYPOINT ["enhanced-rsm"]
CMD ["--help"]
