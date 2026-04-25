package settings

// S holds all tunable parameters for both the Go API and the Python face-service.
// The face-service polls GET /api/settings periodically and applies the relevant fields.
type S struct {
	// ── Recognition (Go matcher) ────────────────────────────────────────────
	MatcherThreshold float64 `json:"matcher_threshold"` // cosine sim to consider same person
	DedupeWindowS    int     `json:"dedupe_window_s"`   // min seconds between recording same face
	EmaAlpha         float64 `json:"ema_alpha"`          // embedding EMA update weight

	// ── Clustering (DBSCAN) ─────────────────────────────────────────────────
	DbscanEps    float64 `json:"dbscan_eps"`     // cosine distance threshold (1-sim)
	DbscanMinPts int     `json:"dbscan_min_pts"` // min cluster size

	// ── Camera & backend ────────────────────────────────────────────────────
	CameraSource string `json:"camera_source"` // "0", "1", "/dev/video0", file path
	FaceBackend  string `json:"face_backend"`  // "onnx" | "mediapipe" | "hailo"
	FrameRotate  int    `json:"frame_rotate"`  // 0 | 90 | 180 | 270

	// ── Detection (Python face-service) ─────────────────────────────────────
	MinConfidence   float64 `json:"min_confidence"`    // detector confidence gate
	CameraFPS       float64 `json:"camera_fps"`        // frames to process per second
	MinFaceSizePx   int     `json:"min_face_size_px"`  // ignore faces smaller than N px wide
	RequireBothEyes bool    `json:"require_both_eyes"` // discard profiles / occluded faces
	MaxYawDeg       float64 `json:"max_yaw_deg"`       // max horizontal head turn (90=disabled)
	MaxPitchDeg     float64 `json:"max_pitch_deg"`     // max vertical head tilt  (90=disabled)
	RequireGaze     bool    `json:"require_gaze"`      // only capture when person looks at camera
	GazeThreshold   float64 `json:"gaze_threshold"`    // min nose-eye symmetry score (0–1)
}

func Default() S {
	return S{
		CameraSource:    "0",
		FaceBackend:     "onnx",
		FrameRotate:     0,
		MatcherThreshold: 0.55,
		DedupeWindowS:    5,
		EmaAlpha:         0.15,
		DbscanEps:        0.42,
		DbscanMinPts:     2,
		MinConfidence:    0.50,
		CameraFPS:        5,
		MinFaceSizePx:    60,
		RequireBothEyes:  false,
		MaxYawDeg:        90,
		MaxPitchDeg:      90,
		RequireGaze:      false,
		GazeThreshold:    0.80,
	}
}
