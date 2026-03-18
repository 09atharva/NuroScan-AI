const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PredictionResult {
    id: number;
    tumor_type: string;
    confidence: number;
    severity: string;
    all_scores: Record<string, number>;
    image_url: string;
    created_at: string;
}

export interface ScanRecord {
    id: number;
    filename: string;
    original_filename: string | null;
    tumor_type: string;
    confidence: number;
    severity: string | null;
    details: Record<string, number> | null;
    created_at: string;
}

export async function uploadScan(file: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
    }

    return response.json();
}

export async function getHistory(limit = 50, offset = 0): Promise<ScanRecord[]> {
    const response = await fetch(
        `${API_BASE}/api/history?limit=${limit}&offset=${offset}`
    );

    if (!response.ok) throw new Error('Failed to fetch history');
    return response.json();
}

export async function getRecord(id: number): Promise<ScanRecord> {
    const response = await fetch(`${API_BASE}/api/history/${id}`);
    if (!response.ok) throw new Error('Record not found');
    return response.json();
}

export async function deleteRecord(id: number): Promise<void> {
    const response = await fetch(`${API_BASE}/api/history/${id}`, {
        method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete record');
}

export function getImageUrl(path: string): string {
    return `${API_BASE}${path}`;
}
