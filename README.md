# FacePulse

Real-time face presence tracking with Prometheus metrics and Grafana dashboards.
Built for Raspberry Pi AI HAT+ (Hailo-10H, 40 TOPS) — runs in Docker on macOS for development.

```
Camera → [Python face-service] → [Go api-service] → Prometheus → Grafana
              InsightFace/Hailo        SQLite + metrics
```

## Features

- Multi-face detection and recognition (ArcFace embeddings, cosine similarity)
- Deduplicated presence logging to SQLite
- Labelling faces via REST API
- Stats per hour / day / month
- Prometheus `/metrics` endpoint
- Grafana dashboard (appearances, unique faces, rates)
- Pluggable backends: ONNX (macOS/Linux CPU) ↔ Hailo-10H (RPi AI HAT+)

## Quick start (macOS)

On macOS, Docker cannot access the camera directly. Run `face-service` on the host:

```bash
# Terminal 1 — API + Prometheus + Grafana in Docker
docker compose up api prometheus grafana --build

# Terminal 2 — face-service on host (uses your webcam)
cd face-service
pip install -r requirements.txt
API_SERVICE_URL=http://localhost:8080 uvicorn main:app --reload
```

Or use the Makefile shortcut:

```bash
make dev
```

Open Grafana at http://localhost:3000 (no login required).

## Quick start (Linux / Raspberry Pi)

On Linux, camera passthrough works in Docker. Uncomment the `devices` section in `docker-compose.yml`, then:

```bash
CAMERA_SOURCE=/dev/video0 docker compose up --build
```

## Labelling faces

```bash
# List detected faces
make faces

# Assign a name
make label FACE_ID=1 LABEL="Alice"

# Stats for the last 24 hours
make stats
```

Or via REST:

```bash
curl -X PUT http://localhost:8080/api/faces/1/label \
     -H "Content-Type: application/json" \
     -d '{"label":"Alice"}'
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/appearance` | Ingest a face embedding (called by face-service) |
| `GET`  | `/api/faces` | List all enrolled faces |
| `PUT`  | `/api/faces/{id}/label` | Set display label |
| `GET`  | `/api/faces/{id}/appearances` | Appearance history |
| `GET`  | `/api/stats?period=hourly\|daily\|monthly&n=N` | Aggregated stats |
| `GET`  | `/metrics` | Prometheus metrics |

## Migrating to Raspberry Pi AI HAT+ (Hailo-10H)

1. Install runtime on RPi OS Bookworm:
   ```bash
   sudo apt install hailo-all
   ```

2. Download models from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo):
   - `scrfd_10g.hef` — face detection
   - `arcface_r50.hef` — face recognition

3. Switch the backend:
   ```bash
   FACE_BACKEND=hailo \
   HAILO_SCRFD_HEF=/models/scrfd_10g.hef \
   HAILO_ARCFACE_HEF=/models/arcface_r50.hef \
   docker compose up
   ```

4. Or build and push ARM images:
   ```bash
   make build-arm
   make deploy-rpi RPI_HOST=raspberrypi.local
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  face-service  (Python / FastAPI)                        │
│  ┌──────────┐   ┌──────────────────────────────────┐    │
│  │  Camera  │──▶│  FaceBackend (onnx / hailo)       │    │
│  │  Loop    │   │  detect → embed → POST /appearance│    │
│  └──────────┘   └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────┐
│  api-service  (Go / stdlib)                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐    │
│  │ Matcher │  │ Storage  │  │ Prometheus /metrics   │    │
│  │ cosine  │  │ SQLite   │  │ appearances_total     │    │
│  │ sim     │  │ WAL mode │  │ unique_faces          │    │
│  └─────────┘  └──────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## License

AGPL-3.0 — see [LICENSE](LICENSE).
