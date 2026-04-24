package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/dzaczek/facepulse/cluster"
	"github.com/dzaczek/facepulse/matcher"
	"github.com/dzaczek/facepulse/metrics"
	"github.com/dzaczek/facepulse/storage"
)

const (
	dedupeWindow    = 5 * time.Second
	embeddingAlpha  = 0.15 // weight of new embedding in EMA update
	dbscanEps       = 0.42 // cosine distance threshold (→ similarity ≥ 0.58)
	dbscanMinPts    = 2
)

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
	Thumbnail  string    `json:"thumbnail"`
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
	mux.HandleFunc("GET /api/suggestions", s.handleSuggestions)
	mux.HandleFunc("POST /api/suggestions/accept", s.handleAcceptSuggestion)
	mux.HandleFunc("POST /api/suggestions/reject", s.handleRejectSuggestion)
	mux.HandleFunc("GET /api/dashboard", s.handleDashboard)
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"status": "ok", "faces": s.matcher.Len()})
	})
}

// ─── Appearance (core detection loop) ────────────────────────────────────────

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

		// Online learning: update stored embedding with EMA
		if !isNew {
			if stored := s.matcher.GetEmbedding(faceID); stored != nil {
				updated := emaEmbedding(stored, req.Embedding, embeddingAlpha)
				s.matcher.Update(faceID, updated)
				_ = s.db.UpdateEmbedding(faceID, updated)
			}
		}
	}

	writeJSON(w, map[string]any{"face_id": faceID, "new": isNew})
}

// ─── Face management ──────────────────────────────────────────────────────────

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

// ─── Suggestions (DBSCAN clustering) ─────────────────────────────────────────

type suggestionResult struct {
	FaceIDs    []int64 `json:"face_ids"`
	Similarity float64 `json:"similarity"`
}

func (s *Server) handleSuggestions(w http.ResponseWriter, r *http.Request) {
	unlabeled, err := s.db.UnlabeledEmbeddings()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	negPairs, _ := s.db.NegativePairs()
	negSet := make(map[[2]int64]bool, len(negPairs))
	for _, p := range negPairs {
		negSet[p] = true
	}

	points := make([]cluster.Point, len(unlabeled))
	for i, f := range unlabeled {
		points[i] = cluster.Point{ID: f.ID, Embedding: f.Embedding}
	}

	clusters := cluster.DBSCAN(points, dbscanEps, dbscanMinPts)

	var suggestions []suggestionResult
	for _, c := range clusters {
		ids := make([]int64, len(c.Points))
		for i, p := range c.Points {
			ids[i] = p.ID
		}
		if allNegative(ids, negSet) {
			continue
		}
		suggestions = append(suggestions, suggestionResult{
			FaceIDs:    ids,
			Similarity: c.AvgSimilarity,
		})
	}

	sort.Slice(suggestions, func(i, j int) bool {
		return suggestions[i].Similarity > suggestions[j].Similarity
	})

	writeJSON(w, suggestions)
}

func (s *Server) handleAcceptSuggestion(w http.ResponseWriter, r *http.Request) {
	var body struct {
		FaceIDs []int64 `json:"face_ids"`
		Label   string  `json:"label"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.FaceIDs) == 0 {
		http.Error(w, "face_ids and label required", http.StatusBadRequest)
		return
	}
	for _, id := range body.FaceIDs {
		if err := s.db.SetLabel(id, body.Label); err != nil {
			log.Printf("accept suggestion label %d: %v", id, err)
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleRejectSuggestion(w http.ResponseWriter, r *http.Request) {
	var body struct {
		FaceIDs []int64 `json:"face_ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	for i := range body.FaceIDs {
		for j := i + 1; j < len(body.FaceIDs); j++ {
			_ = s.db.CreateNegativePair(body.FaceIDs[i], body.FaceIDs[j])
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	timeline, err := s.db.DashboardTimeline()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	top10, err := s.db.DashboardTop10()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	faces, _ := s.db.AllFaces()
	writeJSON(w, map[string]any{
		"total_faces": len(faces),
		"timeline":    timeline,
		"top10":       top10,
	})
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func (s *Server) saveThumbnail(faceID int64, b64 string) {
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return
	}
	dir := filepath.Join(s.dataDir, "thumbs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(dir, fmt.Sprintf("%d.jpg", faceID)), data, 0o644)
}

// emaEmbedding computes: normalize(alpha*new + (1-alpha)*old)
func emaEmbedding(old, new []float64, alpha float64) []float64 {
	if len(old) != len(new) {
		return old
	}
	out := make([]float64, len(old))
	var norm float64
	for i := range old {
		v := (1-alpha)*old[i] + alpha*new[i]
		out[i] = v
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range out {
			out[i] /= norm
		}
	}
	return out
}

func allNegative(ids []int64, negSet map[[2]int64]bool) bool {
	for i := range ids {
		for j := i + 1; j < len(ids); j++ {
			a, b := ids[i], ids[j]
			if a > b {
				a, b = b, a
			}
			if !negSet[[2]int64{a, b}] {
				return false
			}
		}
	}
	return true
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
