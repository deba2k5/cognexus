/**
 * AWE API Service
 * ================
 * Client for communicating with the AWE backend API.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ExplorationRequest {
    url: string;
    objective: string;
    target_fields?: string[];
    max_pages?: number;
    timeout?: number;
}

export interface ExplorationStatus {
    task_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    pages_visited: number;
    items_extracted: number;
    current_url?: string;
    started_at?: string;
    completed_at?: string;
    error?: string;
}

export interface ExplorationResult {
    task_id: string;
    status: string;
    data: Record<string, unknown>[];
    metadata: Record<string, unknown>;
    patterns_learned: number;
    duration_seconds: number;
}

export interface HealthStatus {
    status: string;
    version: string;
    model: string;
    model_provider: string;
    tot_enabled: boolean;
}

export interface DemoResult {
    status: string;
    message: string;
    url: string;
    data: Record<string, unknown>[];
    stats: {
        pages_visited: number;
        items_extracted: number;
        duration_ms: number;
    };
}

class AWEApiClient {
    private baseUrl: string;

    constructor(baseUrl: string = API_URL) {
        this.baseUrl = baseUrl;
    }

    /**
     * Check if the API server is healthy
     */
    async healthCheck(): Promise<HealthStatus> {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Start a new exploration task
     */
    async startExploration(request: ExplorationRequest): Promise<ExplorationStatus> {
        const response = await fetch(`${this.baseUrl}/explore`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Failed to start exploration');
        }

        return response.json();
    }

    /**
     * Get the status of an exploration task
     */
    async getExplorationStatus(taskId: string): Promise<ExplorationStatus> {
        const response = await fetch(`${this.baseUrl}/explore/${taskId}`);

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Task not found');
            }
            throw new Error(`Failed to get status: ${response.statusText}`);
        }

        return response.json();
    }

    /**
     * Get the results of a completed exploration
     */
    async getExplorationResults(taskId: string): Promise<ExplorationResult> {
        const response = await fetch(`${this.baseUrl}/explore/${taskId}/results`);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Failed to get results');
        }

        return response.json();
    }

    /**
     * Run a quick demo exploration
     */
    async runDemo(url: string): Promise<DemoResult> {
        const response = await fetch(`${this.baseUrl}/demo`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url, quick_mode: true }),
        });

        if (!response.ok) {
            throw new Error(`Demo failed: ${response.statusText}`);
        }

        return response.json();
    }

    /**
     * Get server configuration
     */
    async getConfig(): Promise<Record<string, unknown>> {
        const response = await fetch(`${this.baseUrl}/config`);
        if (!response.ok) {
            throw new Error(`Failed to get config: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Poll exploration status until completion
     */
    async waitForCompletion(
        taskId: string,
        onProgress?: (status: ExplorationStatus) => void,
        pollInterval: number = 1000
    ): Promise<ExplorationResult> {
        while (true) {
            const status = await this.getExplorationStatus(taskId);

            if (onProgress) {
                onProgress(status);
            }

            if (status.status === 'completed' || status.status === 'failed') {
                return this.getExplorationResults(taskId);
            }

            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
    }
}

// Export singleton instance
export const aweApi = new AWEApiClient();

// Also export the class for custom configuration
export { AWEApiClient };
