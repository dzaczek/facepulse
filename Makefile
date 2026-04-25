.PHONY: up down dev dev-macos dev-api dev-face build build-arm logs reset label faces stats

# ── Linux / RPi (everything in Docker) ───────────────────────────────────────
up:
	docker-compose up --build -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# ── macOS development (recommended) ──────────────────────────────────────────
# Only postgres + grafana in Docker.
# Go API and face-service run natively — no Docker networking headaches.

# Step 1: start database + grafana
dev-macos:
	docker-compose up postgres grafana -d
	@echo ""
	@echo "✓ postgres  → localhost:5432"
	@echo "✓ grafana   → http://localhost:3000"
	@echo ""
	@echo "Now open TWO more terminals:"
	@echo "  Terminal 2:  make dev-api"
	@echo "  Terminal 3:  make dev-face"
	@echo ""
	@echo "UI → http://localhost:8080   Debug → http://localhost:8000/debug/frame"

# Step 2 (Terminal 2): Go api-service natively
dev-api:
	cd api-service && \
	DATABASE_URL="postgres://facepulse:facepulse@localhost:5432/facepulse?sslmode=disable" \
	DATA_DIR=/tmp/facepulse-data \
	FACE_SERVICE_URL=http://localhost:8000 \
	go run .

# Step 3 (Terminal 3): Python face-service natively (has camera access)
dev-face:
	cd face-service && \
	pip install -r requirements.txt -q && \
	API_SERVICE_URL=http://localhost:8080 \
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Legacy: api in Docker (kept for reference)
dev:
	FACE_SERVICE_URL=http://host.docker.internal:8000 \
		docker-compose up postgres api grafana --build -d
	cd face-service && \
		API_SERVICE_URL=http://localhost:8080 \
		uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# ── Production / RPi ─────────────────────────────────────────────────────────
build-arm:
	docker buildx build --platform linux/arm64 \
		-t ghcr.io/dzaczek/facepulse-api:latest ./api-service --push
	docker buildx build --platform linux/arm64 \
		-t ghcr.io/dzaczek/facepulse-face:latest ./face-service --push

deploy-rpi:
	ssh pi@$(RPI_HOST) 'cd ~/facepulse && docker-compose pull && docker-compose up -d'

reset:
	docker-compose down -v

# ── Helpers ───────────────────────────────────────────────────────────────────
label:
	curl -s -X PUT http://localhost:8080/api/faces/$(FACE_ID)/label \
		-H "Content-Type: application/json" \
		-d '{"label":"$(LABEL)"}' && echo

faces:
	curl -s http://localhost:8080/api/faces | python3 -m json.tool

stats:
	curl -s "http://localhost:8080/api/stats?period=$(or $(PERIOD),hourly)&n=$(or $(N),24)" \
		| python3 -m json.tool
