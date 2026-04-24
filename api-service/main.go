package main

import (
	"embed"
	"io/fs"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/dzaczek/facepulse/api"
	"github.com/dzaczek/facepulse/matcher"
	"github.com/dzaczek/facepulse/metrics"
	"github.com/dzaczek/facepulse/storage"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

//go:embed static
var staticFiles embed.FS

func main() {
	dbPath := env("DB_PATH", "/data/facepulse.db")
	dataDir := env("DATA_DIR", "/data")
	addr := env("LISTEN_ADDR", ":8080")

	db, err := storage.New(dbPath)
	if err != nil {
		log.Fatalf("storage: %v", err)
	}
	defer db.Close()

	m := matcher.New(0)
	if err := loadMatcher(db, m); err != nil {
		log.Printf("matcher initial load: %v", err)
	}

	met := metrics.New()
	faces, _ := db.AllFaces()
	met.UniqueFaces.Set(float64(len(faces)))

	go func() {
		t := time.NewTicker(5 * time.Minute)
		for range t.C {
			if err := loadMatcher(db, m); err != nil {
				log.Printf("matcher refresh: %v", err)
			}
		}
	}()

	staticSub, _ := fs.Sub(staticFiles, "static")

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	srv := api.NewServer(db, m, met, dataDir)
	srv.RegisterRoutes(mux)

	log.Printf("facepulse api listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, cors(mux)))
}

func loadMatcher(db *storage.DB, m *matcher.Matcher) error {
	faces, err := db.AllEmbeddings()
	if err != nil {
		return err
	}
	mf := make([]matcher.Face, len(faces))
	for i, f := range faces {
		mf[i] = matcher.Face{ID: f.ID, Embedding: f.Embedding}
	}
	m.Load(mf)
	log.Printf("matcher loaded %d faces", len(mf))
	return nil
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
