package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type Metrics struct {
	AppearancesTotal *prometheus.CounterVec
	UniqueFaces      prometheus.Gauge
	FramesProcessed  prometheus.Counter
}

func New() *Metrics {
	return &Metrics{
		AppearancesTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "facepulse_appearances_total",
			Help: "Total deduplicated face appearances recorded",
		}, []string{"face_id", "label"}),

		UniqueFaces: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "facepulse_unique_faces",
			Help: "Number of unique faces enrolled in the database",
		}),

		FramesProcessed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "facepulse_frames_processed_total",
			Help: "Total frames received from face-service",
		}),
	}
}
