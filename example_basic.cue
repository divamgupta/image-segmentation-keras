workflow: template & {
	parameter: {
		endpoint: "https://api.example.com"
		uri:      "/jobs"
		method:   "POST"
		body: {name: "my-job"}
		// Uses defaults: pollInterval="5s", maxRetries=30
	}
}
