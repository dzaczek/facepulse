package storage

import (
	"database/sql"
	"encoding/json"
	"time"

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
