package storage

import (
	"database/sql"
	"encoding/json"
	"time"

	"github.com/dzaczek/facepulse/settings"
	_ "modernc.org/sqlite"
)

type DB struct {
	db *sql.DB
}

type Face struct {
	ID        int64     `json:"id"`
	Label     string    `json:"label"`
	CreatedAt time.Time `json:"created_at"`
	Embedding []float64 `json:"-"`
}

type Appearance struct {
	ID         int64     `json:"id"`
	FaceID     int64     `json:"face_id"`
	Confidence float64   `json:"confidence"`
	BBox       []float64 `json:"bbox"`
	SeenAt     time.Time `json:"seen_at"`
}

type StatRow struct {
	Period string `json:"period"`
	FaceID int64  `json:"face_id"`
	Label  string `json:"label"`
	Count  int    `json:"count"`
}

func New(path string) (*DB, error) {
	db, err := sql.Open("sqlite", path+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, err
	}
	if err := migrate(db); err != nil {
		return nil, err
	}
	return &DB{db: db}, nil
}

// TimePoint is used by the dashboard timeline chart.
type TimePoint struct {
	Date  string `json:"date"`
	Count int    `json:"count"`
}

// FaceTop is used by the dashboard top-10 bar chart.
type FaceTop struct {
	FaceID int64  `json:"face_id"`
	Label  string `json:"label"`
	Count  int    `json:"count"`
}

func migrate(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS faces (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			label      TEXT    NOT NULL DEFAULT '',
			embedding  TEXT    NOT NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS appearances (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			face_id    INTEGER NOT NULL REFERENCES faces(id),
			confidence REAL    NOT NULL,
			bbox       TEXT    NOT NULL,
			seen_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS settings (
			id   INTEGER PRIMARY KEY,
			data TEXT    NOT NULL
		);
		CREATE TABLE IF NOT EXISTS negative_pairs (
			face_id_a  INTEGER NOT NULL,
			face_id_b  INTEGER NOT NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (face_id_a, face_id_b)
		);
		CREATE INDEX IF NOT EXISTS idx_appearances_seen_at ON appearances(seen_at);
		CREATE INDEX IF NOT EXISTS idx_appearances_face_id ON appearances(face_id);
	`)
	return err
}

func (d *DB) Close() error { return d.db.Close() }

func (d *DB) AllFaces() ([]Face, error) {
	rows, err := d.db.Query(`SELECT id, label, created_at FROM faces ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var faces []Face
	for rows.Next() {
		var f Face
		if err := rows.Scan(&f.ID, &f.Label, &f.CreatedAt); err != nil {
			return nil, err
		}
		faces = append(faces, f)
	}
	return faces, rows.Err()
}

func (d *DB) AllEmbeddings() ([]Face, error) {
	rows, err := d.db.Query(`SELECT id, label, embedding FROM faces`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var faces []Face
	for rows.Next() {
		var f Face
		var embJSON string
		if err := rows.Scan(&f.ID, &f.Label, &embJSON); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(embJSON), &f.Embedding)
		faces = append(faces, f)
	}
	return faces, rows.Err()
}

func (d *DB) GetFace(id int64) (*Face, error) {
	var f Face
	err := d.db.QueryRow(
		`SELECT id, label, created_at FROM faces WHERE id = ?`, id,
	).Scan(&f.ID, &f.Label, &f.CreatedAt)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func (d *DB) CreateFace(embedding []float64) (int64, error) {
	embJSON, err := json.Marshal(embedding)
	if err != nil {
		return 0, err
	}
	res, err := d.db.Exec(`INSERT INTO faces (embedding) VALUES (?)`, string(embJSON))
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

func (d *DB) SetLabel(faceID int64, label string) error {
	_, err := d.db.Exec(`UPDATE faces SET label = ? WHERE id = ?`, label, faceID)
	return err
}

func (d *DB) RecordAppearance(faceID int64, confidence float64, bbox []float64) error {
	bboxJSON, err := json.Marshal(bbox)
	if err != nil {
		return err
	}
	_, err = d.db.Exec(
		`INSERT INTO appearances (face_id, confidence, bbox) VALUES (?, ?, ?)`,
		faceID, confidence, string(bboxJSON),
	)
	return err
}

func (d *DB) FaceAppearances(faceID int64, limit int) ([]Appearance, error) {
	rows, err := d.db.Query(`
		SELECT id, face_id, confidence, bbox, seen_at
		FROM appearances WHERE face_id = ?
		ORDER BY seen_at DESC LIMIT ?
	`, faceID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var apps []Appearance
	for rows.Next() {
		var a Appearance
		var bboxJSON string
		if err := rows.Scan(&a.ID, &a.FaceID, &a.Confidence, &bboxJSON, &a.SeenAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(bboxJSON), &a.BBox)
		apps = append(apps, a)
	}
	return apps, rows.Err()
}

func (d *DB) StatsHourly(hours int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT strftime('%Y-%m-%d %H:00', a.seen_at) as period,
		       a.face_id, COALESCE(NULLIF(f.label,''), CAST(a.face_id AS TEXT)) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= datetime('now', ? || ' hours')
		GROUP BY period, a.face_id
		ORDER BY period DESC, count DESC
	`, -hours)
}

func (d *DB) StatsDaily(days int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT strftime('%Y-%m-%d', a.seen_at) as period,
		       a.face_id, COALESCE(NULLIF(f.label,''), CAST(a.face_id AS TEXT)) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= datetime('now', ? || ' days')
		GROUP BY period, a.face_id
		ORDER BY period DESC, count DESC
	`, -days)
}

func (d *DB) StatsMonthly(months int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT strftime('%Y-%m', a.seen_at) as period,
		       a.face_id, COALESCE(NULLIF(f.label,''), CAST(a.face_id AS TEXT)) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= datetime('now', ? || ' months')
		GROUP BY period, a.face_id
		ORDER BY period DESC, count DESC
	`, -months)
}

func (d *DB) queryStats(query string, arg interface{}) ([]StatRow, error) {
	rows, err := d.db.Query(query, arg)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var stats []StatRow
	for rows.Next() {
		var s StatRow
		if err := rows.Scan(&s.Period, &s.FaceID, &s.Label, &s.Count); err != nil {
			return nil, err
		}
		stats = append(stats, s)
	}
	return stats, rows.Err()
}

func (d *DB) DeleteFace(id int64) error {
	if _, err := d.db.Exec(`DELETE FROM appearances WHERE face_id = ?`, id); err != nil {
		return err
	}
	if _, err := d.db.Exec(`DELETE FROM negative_pairs WHERE face_id_a = ? OR face_id_b = ?`, id, id); err != nil {
		return err
	}
	_, err := d.db.Exec(`DELETE FROM faces WHERE id = ?`, id)
	return err
}

// ─── Learning / Clustering ────────────────────────────────────────────────────

func (d *DB) UnlabeledEmbeddings() ([]Face, error) {
	rows, err := d.db.Query(
		`SELECT id, embedding FROM faces WHERE label = '' OR label IS NULL`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var faces []Face
	for rows.Next() {
		var f Face
		var embJSON string
		if err := rows.Scan(&f.ID, &embJSON); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(embJSON), &f.Embedding)
		faces = append(faces, f)
	}
	return faces, rows.Err()
}

func (d *DB) GetEmbedding(id int64) ([]float64, error) {
	var embJSON string
	err := d.db.QueryRow(`SELECT embedding FROM faces WHERE id = ?`, id).Scan(&embJSON)
	if err != nil {
		return nil, err
	}
	var emb []float64
	return emb, json.Unmarshal([]byte(embJSON), &emb)
}

func (d *DB) UpdateEmbedding(id int64, emb []float64) error {
	embJSON, err := json.Marshal(emb)
	if err != nil {
		return err
	}
	_, err = d.db.Exec(`UPDATE faces SET embedding = ? WHERE id = ?`, string(embJSON), id)
	return err
}

func (d *DB) CreateNegativePair(a, b int64) error {
	lo, hi := a, b
	if lo > hi {
		lo, hi = hi, lo
	}
	_, err := d.db.Exec(
		`INSERT OR IGNORE INTO negative_pairs (face_id_a, face_id_b) VALUES (?, ?)`, lo, hi,
	)
	return err
}

func (d *DB) NegativePairs() ([][2]int64, error) {
	rows, err := d.db.Query(`SELECT face_id_a, face_id_b FROM negative_pairs`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var pairs [][2]int64
	for rows.Next() {
		var p [2]int64
		if err := rows.Scan(&p[0], &p[1]); err != nil {
			return nil, err
		}
		pairs = append(pairs, p)
	}
	return pairs, rows.Err()
}

// ─── Settings ────────────────────────────────────────────────────────────────

func (d *DB) GetSettings() (settings.S, error) {
	var data string
	err := d.db.QueryRow(`SELECT data FROM settings WHERE id = 1`).Scan(&data)
	if err == sql.ErrNoRows {
		return settings.Default(), nil
	}
	if err != nil {
		return settings.Default(), err
	}
	s := settings.Default()
	return s, json.Unmarshal([]byte(data), &s)
}

func (d *DB) SaveSettings(s settings.S) error {
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	_, err = d.db.Exec(
		`INSERT INTO settings (id, data) VALUES (1, ?)
		 ON CONFLICT(id) DO UPDATE SET data = excluded.data`,
		string(data),
	)
	return err
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

func (d *DB) DashboardTimeline() ([]TimePoint, error) {
	rows, err := d.db.Query(`
		SELECT date(created_at) as date, COUNT(*) as cnt
		FROM faces
		GROUP BY date(created_at)
		ORDER BY date
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var pts []TimePoint
	for rows.Next() {
		var p TimePoint
		if err := rows.Scan(&p.Date, &p.Count); err != nil {
			return nil, err
		}
		pts = append(pts, p)
	}
	return pts, rows.Err()
}

func (d *DB) DashboardTop10() ([]FaceTop, error) {
	// Group by label so all face_ids for "Jacek" are summed together.
	// Unlabelled faces fall back to their individual IDs.
	rows, err := d.db.Query(`
		SELECT MIN(f.id) as face_id,
		       COALESCE(NULLIF(f.label,''), '#' || CAST(f.id AS TEXT)) as label,
		       COUNT(a.id) as cnt
		FROM faces f
		LEFT JOIN appearances a ON a.face_id = f.id
		GROUP BY COALESCE(NULLIF(f.label,''), '#' || CAST(f.id AS TEXT))
		ORDER BY cnt DESC
		LIMIT 10
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tops []FaceTop
	for rows.Next() {
		var t FaceTop
		if err := rows.Scan(&t.FaceID, &t.Label, &t.Count); err != nil {
			return nil, err
		}
		tops = append(tops, t)
	}
	return tops, rows.Err()
}
