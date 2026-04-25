package storage

import (
	"database/sql"
	"encoding/json"
	"time"

	"github.com/dzaczek/facepulse/settings"
	_ "github.com/jackc/pgx/v5/stdlib"
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

type TimePoint struct {
	Date  string `json:"date"`
	Count int    `json:"count"`
}

type FaceTop struct {
	FaceID int64  `json:"face_id"`
	Label  string `json:"label"`
	Count  int    `json:"count"`
}

func New(connStr string) (*DB, error) {
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(20)
	db.SetMaxIdleConns(5)
	if err := migrate(db); err != nil {
		return nil, err
	}
	return &DB{db: db}, nil
}

func migrate(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS faces (
			id         BIGSERIAL    PRIMARY KEY,
			label      TEXT         NOT NULL DEFAULT '',
			embedding  TEXT         NOT NULL,
			created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
		);
		CREATE TABLE IF NOT EXISTS appearances (
			id         BIGSERIAL    PRIMARY KEY,
			face_id    BIGINT       NOT NULL REFERENCES faces(id),
			confidence REAL         NOT NULL,
			bbox       TEXT         NOT NULL,
			seen_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
		);
		CREATE TABLE IF NOT EXISTS settings (
			id   INTEGER PRIMARY KEY,
			data TEXT    NOT NULL
		);
		CREATE TABLE IF NOT EXISTS negative_pairs (
			face_id_a  BIGINT      NOT NULL,
			face_id_b  BIGINT      NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY (face_id_a, face_id_b)
		);
		CREATE INDEX IF NOT EXISTS idx_appearances_seen_at ON appearances(seen_at);
		CREATE INDEX IF NOT EXISTS idx_appearances_face_id ON appearances(face_id);
	`)
	return err
}

func (d *DB) Close() error { return d.db.Close() }

// ─── Faces ────────────────────────────────────────────────────────────────────

func (d *DB) AllFaces() ([]Face, error) {
	rows, err := d.db.Query(`SELECT id, label, created_at FROM faces ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Face
	for rows.Next() {
		var f Face
		if err := rows.Scan(&f.ID, &f.Label, &f.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

func (d *DB) AllEmbeddings() ([]Face, error) {
	rows, err := d.db.Query(`SELECT id, label, embedding FROM faces`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Face
	for rows.Next() {
		var f Face
		var js string
		if err := rows.Scan(&f.ID, &f.Label, &js); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(js), &f.Embedding)
		out = append(out, f)
	}
	return out, rows.Err()
}

func (d *DB) GetFace(id int64) (*Face, error) {
	var f Face
	err := d.db.QueryRow(
		`SELECT id, label, created_at FROM faces WHERE id = $1`, id,
	).Scan(&f.ID, &f.Label, &f.CreatedAt)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func (d *DB) CreateFace(embedding []float64) (int64, error) {
	js, err := json.Marshal(embedding)
	if err != nil {
		return 0, err
	}
	var id int64
	err = d.db.QueryRow(
		`INSERT INTO faces (embedding) VALUES ($1) RETURNING id`, string(js),
	).Scan(&id)
	return id, err
}

func (d *DB) SetLabel(faceID int64, label string) error {
	_, err := d.db.Exec(`UPDATE faces SET label = $1 WHERE id = $2`, label, faceID)
	return err
}

func (d *DB) DeleteFace(id int64) error {
	if _, err := d.db.Exec(`DELETE FROM appearances WHERE face_id = $1`, id); err != nil {
		return err
	}
	if _, err := d.db.Exec(`DELETE FROM negative_pairs WHERE face_id_a = $1 OR face_id_b = $1`, id, id); err != nil {
		return err
	}
	_, err := d.db.Exec(`DELETE FROM faces WHERE id = $1`, id)
	return err
}

func (d *DB) GetEmbedding(id int64) ([]float64, error) {
	var js string
	if err := d.db.QueryRow(`SELECT embedding FROM faces WHERE id = $1`, id).Scan(&js); err != nil {
		return nil, err
	}
	var emb []float64
	return emb, json.Unmarshal([]byte(js), &emb)
}

func (d *DB) UpdateEmbedding(id int64, emb []float64) error {
	js, err := json.Marshal(emb)
	if err != nil {
		return err
	}
	_, err = d.db.Exec(`UPDATE faces SET embedding = $1 WHERE id = $2`, string(js), id)
	return err
}

// ─── Appearances ──────────────────────────────────────────────────────────────

func (d *DB) RecordAppearance(faceID int64, confidence float64, bbox []float64) error {
	js, err := json.Marshal(bbox)
	if err != nil {
		return err
	}
	_, err = d.db.Exec(
		`INSERT INTO appearances (face_id, confidence, bbox) VALUES ($1, $2, $3)`,
		faceID, confidence, string(js),
	)
	return err
}

func (d *DB) FaceAppearances(faceID int64, limit int) ([]Appearance, error) {
	rows, err := d.db.Query(`
		SELECT id, face_id, confidence, bbox, seen_at
		FROM appearances WHERE face_id = $1
		ORDER BY seen_at DESC LIMIT $2
	`, faceID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Appearance
	for rows.Next() {
		var a Appearance
		var js string
		if err := rows.Scan(&a.ID, &a.FaceID, &a.Confidence, &js, &a.SeenAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(js), &a.BBox)
		out = append(out, a)
	}
	return out, rows.Err()
}

// ─── Stats ────────────────────────────────────────────────────────────────────

func (d *DB) StatsHourly(hours int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT TO_CHAR(DATE_TRUNC('hour', a.seen_at), 'YYYY-MM-DD HH24:00') as period,
		       a.face_id,
		       COALESCE(NULLIF(f.label,''), f.id::text) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= NOW() - $1 * INTERVAL '1 hour'
		GROUP BY period, a.face_id, f.label, f.id
		ORDER BY period DESC, count DESC
	`, hours)
}

func (d *DB) StatsDaily(days int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT TO_CHAR(DATE_TRUNC('day', a.seen_at), 'YYYY-MM-DD') as period,
		       a.face_id,
		       COALESCE(NULLIF(f.label,''), f.id::text) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= NOW() - $1 * INTERVAL '1 day'
		GROUP BY period, a.face_id, f.label, f.id
		ORDER BY period DESC, count DESC
	`, days)
}

func (d *DB) StatsMonthly(months int) ([]StatRow, error) {
	return d.queryStats(`
		SELECT TO_CHAR(DATE_TRUNC('month', a.seen_at), 'YYYY-MM') as period,
		       a.face_id,
		       COALESCE(NULLIF(f.label,''), f.id::text) as label,
		       COUNT(*) as count
		FROM appearances a JOIN faces f ON f.id = a.face_id
		WHERE a.seen_at >= NOW() - $1 * INTERVAL '1 month'
		GROUP BY period, a.face_id, f.label, f.id
		ORDER BY period DESC, count DESC
	`, months)
}

func (d *DB) queryStats(query string, arg interface{}) ([]StatRow, error) {
	rows, err := d.db.Query(query, arg)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []StatRow
	for rows.Next() {
		var s StatRow
		if err := rows.Scan(&s.Period, &s.FaceID, &s.Label, &s.Count); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

func (d *DB) DashboardTimeline() ([]TimePoint, error) {
	rows, err := d.db.Query(`
		SELECT TO_CHAR(DATE_TRUNC('day', created_at), 'YYYY-MM-DD') as date,
		       COUNT(*) as cnt
		FROM faces
		GROUP BY DATE_TRUNC('day', created_at)
		ORDER BY date
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []TimePoint
	for rows.Next() {
		var p TimePoint
		if err := rows.Scan(&p.Date, &p.Count); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (d *DB) DashboardTop10() ([]FaceTop, error) {
	rows, err := d.db.Query(`
		SELECT MIN(f.id) as face_id,
		       COALESCE(NULLIF(f.label,''), '#' || f.id::text) as label,
		       COUNT(a.id) as cnt
		FROM faces f
		LEFT JOIN appearances a ON a.face_id = f.id
		GROUP BY COALESCE(NULLIF(f.label,''), '#' || f.id::text)
		ORDER BY cnt DESC
		LIMIT 10
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []FaceTop
	for rows.Next() {
		var t FaceTop
		if err := rows.Scan(&t.FaceID, &t.Label, &t.Count); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// ─── Clustering ───────────────────────────────────────────────────────────────

func (d *DB) UnlabeledEmbeddings() ([]Face, error) {
	rows, err := d.db.Query(
		`SELECT id, embedding FROM faces WHERE label = '' OR label IS NULL`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Face
	for rows.Next() {
		var f Face
		var js string
		if err := rows.Scan(&f.ID, &js); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(js), &f.Embedding)
		out = append(out, f)
	}
	return out, rows.Err()
}

func (d *DB) CreateNegativePair(a, b int64) error {
	lo, hi := a, b
	if lo > hi {
		lo, hi = hi, lo
	}
	_, err := d.db.Exec(
		`INSERT INTO negative_pairs (face_id_a, face_id_b) VALUES ($1, $2)
		 ON CONFLICT (face_id_a, face_id_b) DO NOTHING`,
		lo, hi,
	)
	return err
}

func (d *DB) NegativePairs() ([][2]int64, error) {
	rows, err := d.db.Query(`SELECT face_id_a, face_id_b FROM negative_pairs`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out [][2]int64
	for rows.Next() {
		var p [2]int64
		if err := rows.Scan(&p[0], &p[1]); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ─── Settings ─────────────────────────────────────────────────────────────────

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
		`INSERT INTO settings (id, data) VALUES (1, $1)
		 ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data`,
		string(data),
	)
	return err
}
