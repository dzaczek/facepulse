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
