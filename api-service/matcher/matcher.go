package matcher

import (
	"math"
	"sync"
)

const defaultThreshold = 0.55

type Face struct {
	ID        int64
	Embedding []float64
}

type Matcher struct {
	mu        sync.RWMutex
	faces     []Face
	threshold float64
}

func New(threshold float64) *Matcher {
	if threshold == 0 {
		threshold = defaultThreshold
	}
	return &Matcher{threshold: threshold}
}

func (m *Matcher) Load(faces []Face) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.faces = make([]Face, len(faces))
	copy(m.faces, faces)
}

func (m *Matcher) Add(f Face) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.faces = append(m.faces, f)
}

func (m *Matcher) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.faces)
}

// Match returns best matching face ID and similarity score.
// Returns -1 if no face exceeds the threshold.
// Remove deletes a face from the in-memory index.
func (m *Matcher) Remove(id int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, f := range m.faces {
		if f.ID == id {
			m.faces = append(m.faces[:i], m.faces[i+1:]...)
			return
		}
	}
}

// Update replaces the stored embedding for an existing face (online learning).
func (m *Matcher) Update(id int64, embedding []float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range m.faces {
		if m.faces[i].ID == id {
			m.faces[i].Embedding = embedding
			return
		}
	}
}

// GetEmbedding returns the current in-memory embedding for a face, or nil if not found.
func (m *Matcher) GetEmbedding(id int64) []float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, f := range m.faces {
		if f.ID == id {
			return f.Embedding
		}
	}
	return nil
}

// Match returns best matching face ID and similarity score.
// Returns -1 if no face exceeds the threshold.
func (m *Matcher) Match(embedding []float64) (int64, float64) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	bestID := int64(-1)
	bestSim := 0.0

	for _, f := range m.faces {
		sim := cosine(embedding, f.Embedding)
		if sim > bestSim {
			bestSim = sim
			bestID = f.ID
		}
	}

	if bestSim < m.threshold {
		return -1, bestSim
	}
	return bestID, bestSim
}

func cosine(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
