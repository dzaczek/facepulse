package cluster

import "math"

type Point struct {
	ID        int64
	Embedding []float64
}

type Cluster struct {
	Points     []Point
	AvgSimilarity float64
}

const (
	unvisited = 0
	noise     = -1
)

// DBSCAN clusters points by cosine similarity.
// eps is the cosine *distance* threshold (1 - similarity), e.g. 0.45 → similarity ≥ 0.55.
func DBSCAN(points []Point, eps float64, minPts int) []Cluster {
	n := len(points)
	labels := make([]int, n)
	clusterID := 0

	for i := range points {
		if labels[i] != unvisited {
			continue
		}
		nb := regionQuery(points, i, eps)
		if len(nb) < minPts {
			labels[i] = noise
			continue
		}
		clusterID++
		labels[i] = clusterID
		seed := append([]int(nil), nb...)
		for j := 0; j < len(seed); j++ {
			idx := seed[j]
			if labels[idx] == noise {
				labels[idx] = clusterID
			}
			if labels[idx] != unvisited {
				continue
			}
			labels[idx] = clusterID
			nn := regionQuery(points, idx, eps)
			if len(nn) >= minPts {
				seed = append(seed, nn...)
			}
		}
	}

	clusterMap := map[int][]Point{}
	for i, label := range labels {
		if label > 0 {
			clusterMap[label] = append(clusterMap[label], points[i])
		}
	}

	result := make([]Cluster, 0, len(clusterMap))
	for _, pts := range clusterMap {
		result = append(result, Cluster{
			Points:        pts,
			AvgSimilarity: avgPairwiseSimilarity(pts),
		})
	}
	return result
}

// AverageEmbedding returns the normalised mean of all point embeddings.
func AverageEmbedding(pts []Point) []float64 {
	if len(pts) == 0 {
		return nil
	}
	dim := len(pts[0].Embedding)
	avg := make([]float64, dim)
	for _, p := range pts {
		for i, v := range p.Embedding {
			avg[i] += v
		}
	}
	for i := range avg {
		avg[i] /= float64(len(pts))
	}
	return normalize(avg)
}

func regionQuery(points []Point, idx int, eps float64) []int {
	var out []int
	for i := range points {
		if i != idx && cosineDist(points[idx].Embedding, points[i].Embedding) <= eps {
			out = append(out, i)
		}
	}
	return out
}

func avgPairwiseSimilarity(pts []Point) float64 {
	if len(pts) < 2 {
		return 1
	}
	var sum float64
	var n int
	for i := range pts {
		for j := i + 1; j < len(pts); j++ {
			sum += 1 - cosineDist(pts[i].Embedding, pts[j].Embedding)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func cosineDist(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1
	}
	return 1 - dot/(math.Sqrt(normA)*math.Sqrt(normB))
}

func normalize(v []float64) []float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}
