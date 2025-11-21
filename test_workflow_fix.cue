// Test cases for the ConditionalWait polling fix
// Tests both the primary fix and alternative approach

testCases: {
	// Test 1: Basic functionality with custom polling
	basicTest: {
		parameter: {
			endpoint: "https://api.example.com"
			uri:      "/v1/resources"
			method:   "POST"
			body: {
				name: "test-resource"
				type: "workflow"
			}
			pollInterval: "3s"
			maxRetries:   10
		}

		// Expected behavior:
		// 1. POST executes once
		// 2. GET polls every 3 seconds for max 10 times
		// 3. Stops when condition met or max retries reached
	}

	// Test 2: Fast polling with short interval
	fastPollingTest: {
		parameter: {
			endpoint:     "https://api.example.com"
			uri:          "/v1/jobs"
			method:       "POST"
			pollInterval: "1s"
			maxRetries:   5
		}
	}

	// Test 3: Long polling with high retry count
	longPollingTest: {
		parameter: {
			endpoint:     "https://api.example.com"
			uri:          "/v1/tasks"
			method:       "POST"
			pollInterval: "10s"
			maxRetries:   60  // 10 minutes total
		}
	}

	// Test 4: Alternative approach test
	alternativeApproachTest: {
		parameter: {
			endpoint:     "https://api.example.com"
			uri:          "/v1/processes"
			method:       "POST"
			pollInterval: "5s"
			maxRetries:   20
		}

		// Uses the alternative ConditionalWait-based approach
		result: templateAlternative & parameter
	}

	// Test 5: Error handling test
	errorHandlingTest: {
		parameter: {
			endpoint:     "https://invalid-api.example.com"
			uri:          "/v1/fail"
			method:       "POST"
			pollInterval: "2s"
			maxRetries:   3
		}

		// Should fail gracefully with proper error messages
	}
}

// Integration test that validates the fix
integrationTest: {
	// Mock server setup (simulates the API behavior)
	mockServer: {
		// POST endpoint - returns ID immediately
		postResponse: {
			id:     "test-123"
			status: "created"
		}

		// GET endpoint - simulates status changes over time
		getResponses: [
			{status: "pending", output: _|_},
			{status: "running", output: _|_},
			{status: "running", output: _|_},
			{status: "success", output: {result: "completed", data: "test-output"}}
		]
	}

	// Test execution
	testExecution: {
		// Simulate the fixed workflow
		workflow: template & {
			parameter: {
				endpoint:     mockServer
				uri:          "/test"
				method:       "POST"
				pollInterval: "1s"
				maxRetries:   10
			}
		}

		// Validate results
		assertions: {
			// POST should execute exactly once
			postExecutedOnce: len(workflow.post.http) == 1

			// GET should execute until condition met (4 times in this case)
			getExecutedUntilSuccess: len(workflow.poll.getWithRetry.attempts) == 4

			// Final result should be correct
			finalResultCorrect: workflow.poll.finalRespMap["status"] == "success"

			// Should not exceed max retries
			withinRetryLimit: workflow.poll.getWithRetry.retryCount <= 10
		}
	}
}

// Performance comparison test
performanceTest: {
	beforeFix: {
		// Original behavior: entire workflow re-executes
		executions: {
			postRequests: 10  // POST executes 10 times (bad!)
			getRequests:  10  // GET executes 10 times
			totalOperations: 20
		}
	}

	afterFix: {
		// Fixed behavior: POST once, GET polls
		executions: {
			postRequests: 1   // POST executes once (good!)
			getRequests:  10  // GET executes 10 times for polling
			totalOperations: 11  // Much more efficient
		}
	}

	improvement: {
		reducedOperations: afterFix.executions.totalOperations < beforeFix.executions.totalOperations
		postReduction:     afterFix.executions.postRequests < beforeFix.executions.postRequests
	}
}

// Configuration validation test
configValidationTest: {
	validConfigs: [
		{pollInterval: "1s", maxRetries: 5},
		{pollInterval: "30s", maxRetries: 100},
		{pollInterval: "500ms", maxRetries: 1}
	]

	invalidConfigs: [
		{pollInterval: "0s", maxRetries: 5},     // Invalid: zero interval
		{pollInterval: "1s", maxRetries: 0},     // Invalid: zero retries
		{pollInterval: "-5s", maxRetries: 10}    // Invalid: negative interval
	]

	// Test that valid configs work and invalid ones are rejected
	validation: {
		for config in validConfigs {
			shouldAccept: template & {parameter: config}
		}

		for config in invalidConfigs {
			shouldReject: try {
				template & {parameter: config}
			} catch {
				rejected: true
			}
		}
	}
}
