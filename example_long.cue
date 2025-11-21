workflow: template & {
	parameter: {
		endpoint:     "https://api.example.com"
		uri:          "/slow-jobs"
		method:       "POST"
		pollInterval: "30s"   // Poll every 30 seconds
		maxRetries:   120     // Max 1 hour total
	}
}
