workflow: template & {
	parameter: {
		endpoint:     "https://api.example.com"
		uri:          "/quick-jobs"
		method:       "POST"
		pollInterval: "1s"    // Poll every second
		maxRetries:   30      // Max 30 seconds total
	}
}
