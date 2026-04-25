.PHONY: up down dev build build-arm push logs reset

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

# macOS dev: face-service runs natively (camera), api in Docker with host.docker.internal
dev:
	FACE_SERVICE_URL=http://host.docker.internal:8000 \
		docker compose up postgres api grafana --build -d
	@echo "--- Starting face-service natively (macOS camera) ---"
	cd face-service && pip install -r requirements.txt && \
		API_SERVICE_URL=http://localhost:8080 uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Build multi-arch images for RPi (linux/arm64)
build-arm:
	docker buildx build --platform linux/arm64 \
		-t ghcr.io/dzaczek/facepulse-api:latest ./api-service --push
	docker buildx build --platform linux/arm64 \
		-t ghcr.io/dzaczek/facepulse-face:latest ./face-service --push

# Deploy to RPi over SSH
deploy-rpi:
	ssh pi@$(RPI_HOST) 'cd ~/facepulse && docker compose pull && docker compose up -d'

reset:
	docker compose down -v

# Assign a label to a face: make label FACE_ID=1 LABEL="Alice"
label:
	curl -s -X PUT http://localhost:8080/api/faces/$(FACE_ID)/label \
		-H "Content-Type: application/json" \
		-d '{"label":"$(LABEL)"}' && echo

faces:
	curl -s http://localhost:8080/api/faces | python3 -m json.tool

stats:
	curl -s "http://localhost:8080/api/stats?period=$(or $(PERIOD),hourly)&n=$(or $(N),24)" | python3 -m json.tool
