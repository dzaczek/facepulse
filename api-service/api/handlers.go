package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/dzaczek/facepulse/matcher"
	"github.com/dzaczek/facepulse/metrics"
	"github.com/dzaczek/facepulse/storage"
)

const dedupeWindow = 5 * time.Second

type Server struct {
	db      *storage.DB
	matcher *matcher.Matcher
	metrics *metrics.Metrics
	dataDir string

	mu       sync.Mutex
	lastSeen map[int64]time.Time
}

type appearanceReq struct {
	Embedding  []float64 `json:"embedding"`
	BBox       []float64 `json:"bbox"`
	Confidence float64   `json:"confidence"`
	Thumbnail  string    `json:"thumbnail"` // base64 JPEG, optional
}

func NewServer(db *storage.DB, m *matcher.Matcher, met *metrics.Metrics, dataDir string) *Server {
	return &Server{
		db:       db,
		matcher:  m,
		metrics:  met,
		dataDir:  dataDir,
		lastSeen: make(map[int64]time.Time),
	}
}

func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/appearance", s.handleAppearance)
	mux.HandleFunc("GET /api/faces", s.handleListFaces)
	mux.HandleFunc("PUT /api/faces/{id}/label", s.handleSetLabel)
	mux.HandleFunc("DELETE /api/faces/{id}/label", s.handleClearLabel)
	mux.HandleFunc("GET /api/faces/{id}/appearances", s.handleFaceAppearances)
	mux.HandleFunc("GET /api/faces/{id}/thumb", s.handleThumb)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"status": "ok", "faces": s.matcher.Len()})
	})
}

func (s *Server) handleAppearance(w http.ResponseWriter, r *http.Request) {
	s.metrics.FramesProcessed.Inc()

	var req appearanceReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if len(req.Embedding) == 0 {
		http.Error(w, "embedding required", http.StatusBadRequest)
		return
	}

	faceID, _ := s.matcher.Match(req.Embedding)
	isNew := faceID == -1

	if isNew {
		var err error
		faceID, err = s.db.CreateFace(req.Embedding)
		if err != nil {
			log.Printf("create face: %v", err)
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		s.matcher.Add(matcher.Face{ID: faceID, Embedding: req.Embedding})
		s.metrics.UniqueFaces.Inc()
		log.Printf("new face enrolled: id=%d", faceID)

		// Save thumbnail on first detection
		if req.Thumbnail != "" {
			s.saveThumbnail(faceID, req.Thumbnail)
		}
	}

	s.mu.Lock()
	last, exists := s.lastSeen[faceID]
	shouldRecord := !exists || time.Since(last) > dedupeWindow
	if shouldRecord {
		s.lastSeen[faceID] = time.Now()
	}
	s.mu.Unlock()

	if shouldRecord {
		if err := s.db.RecordAppearance(faceID, req.Confidence, req.BBox); err != nil {
			log.Printf("record appearance: %v", err)
		}
		label := ""
		if f, err := s.db.GetFace(faceID); err == nil {
			label = f.Label
		}
		s.metrics.AppearancesTotal.WithLabelValues(
			strconv.FormatInt(faceID, 10), label,
		).Inc()
	}

	writeJSON(w, map[string]any{"face_id": faceID, "new": isNew})
}

func (s *Server) handleListFaces(w http.ResponseWriter, r *http.Request) {
	faces, err := s.db.AllFaces()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, faces)
}

func (s *Server) handleSetLabel(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	var body struct {
		Label string `json:"label"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.db.SetLabel(id, body.Label); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleClearLabel(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	if err := s.db.SetLabel(id, ""); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleThumb(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	path := filepath.Join(s.dataDir, "thumbs", fmt.Sprintf("%d.jpg", id))
	if _, err := os.Stat(path); os.IsNotExist(err) {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Cache-Control", "public, max-age=86400")
	http.ServeFile(w, r, path)
}

func (s *Server) handleFaceAppearances(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		http.Error(w, "invalid id", http.StatusBadRequest)
		return
	}
	limit := 100
	if l := r.URL.Query().Get("limit"); l != "" {
		if n, err := strconv.Atoi(l); err == nil && n > 0 {
			limit = n
		}
	}
	apps, err := s.db.FaceAppearances(id, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, apps)
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	period := r.URL.Query().Get("period")
	n, _ := strconv.Atoi(r.URL.Query().Get("n"))
	if n <= 0 {
		n = 24
	}
	var rows []storage.StatRow
	var err error
	switch period {
	case "daily":
		rows, err = s.db.StatsDaily(n)
	case "monthly":
		rows, err = s.db.StatsMonthly(n)
	default:
		rows, err = s.db.StatsHourly(n)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, rows)
}

func (s *Server) saveThumbnail(faceID int64, b64 string) {
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return
	}
	dir := filepath.Join(s.dataDir, "thumbs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("thumbs dir: %v", err)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("%d.jpg", faceID))
	if err := os.WriteFile(path, data, 0o644); err != nil {
		log.Printf("save thumb %d: %v", faceID, err)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
