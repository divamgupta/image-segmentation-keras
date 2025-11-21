template: {
	// Parameters with custom polling options
	parameter: {
		endpoint: string
		uri:      string
		method:   string
		body?:    {...}
		header?:  {...}

		// NEW: Custom polling configuration
		pollInterval: *"5s" | string  // Default 5 seconds
		maxRetries:   *30 | int       // Default 30 retries
	}

	// Step 1: Execute POST request ONCE
	post: op.#Steps & {
		parts: ["(parameter.endpoint)", "(parameter.uri)"]
		accessUrl: strings.Join(parts, "")

		http: op.#HTTPDo & {
			method: parameter.method
			url: accessUrl
			request: {
				if parameter.body != _|_ {
					body: json.Marshal(parameter.body)
				}
				if parameter.header != _|_ {
					header: parameter.header
				}
				timeout: "10s"
			}
		}

		postValidation: op.#Steps & {
			if http.response.statusCode > 299 {
				fail: op.#Fail & {
					message: "POST request failed: \(http.response.statusCode)"
				}
			}
		}

		httpRespMap: json.Unmarshal(http.response.body)
		postId:      httpRespMap["id"]
	}

	// Step 2: Poll GET request with CUSTOM SETTINGS
	poll: op.#Steps & {
		getParts: ["(parameter.endpoint)", "(parameter.uri)", "/", "\(post.postId)"]
		getUrl: strings.Join(getParts, "")

		// NEW COMPONENT: HTTPGetWithRetry
		getWithRetry: op.#HTTPGetWithRetry & {
			url: getUrl
			request: {
				header: {"Content-Type": "application/json"}
				rateLimiter: {limit: 200, period: "5s"}
			}

			// CUSTOM POLLING CONFIGURATION - This solves the core issue!
			retry: {
				maxAttempts: parameter.maxRetries
				interval:    parameter.pollInterval
			}

			// SUCCESS CONDITION
			continueCondition: {
				respMap: json.Unmarshal(response.body)
				shouldContinue: !(respMap["status"] == "success" && respMap["output"] != _|_)
			}
		}

		getValidation: op.#Steps & {
			if getWithRetry.response.statusCode > 200 {
				fail: op.#Fail & {
					message: "GET request failed after \(parameter.maxRetries) retries"
				}
			}
		}

		finalRespMap: json.Unmarshal(getWithRetry.response.body)
	}

	// Step 3: Output results
	output: op.#Steps & {
		result: {
			data:   poll.finalRespMap["output"]
			status: poll.finalRespMap["status"]
			postId: post.postId
			totalRetries: poll.getWithRetry.retryCount
			duration:     poll.getWithRetry.totalDuration
		}
	}
}

// Required component definition
#HTTPGetWithRetry: {
	url: string
	request: #HTTPRequest
	retry: {
		maxAttempts: int
		interval: string
	}
	continueCondition: {
		shouldContinue: bool
	}
	response: #HTTPResponse
	retryCount: int
	totalDuration: string
}
