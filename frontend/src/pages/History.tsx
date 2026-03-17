import { useState, useEffect, useCallback } from 'react';
import { Trash2, RefreshCw, Inbox } from 'lucide-react';
import { getHistory, deleteRecord, type ScanRecord } from '../api/client';

export default function HistoryPage() {
    const [records, setRecords] = useState<ScanRecord[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchHistory = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getHistory();
            setRecords(data);
        } catch {
            setError('Failed to load history. Is the backend running?');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchHistory();
    }, [fetchHistory]);

    const handleDelete = async (id: number) => {
        if (!confirm('Delete this scan record?')) return;
        try {
            await deleteRecord(id);
            setRecords((prev) => prev.filter((r) => r.id !== id));
        } catch {
            setError('Failed to delete record.');
        }
    };

    const getSeverityClass = (severity: string | null) => {
        if (!severity) return 'none';
        return severity.toLowerCase() as 'high' | 'moderate' | 'low' | 'none';
    };

    return (
        <div className="history-page container">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <div>
                    <h1>Scan History</h1>
                    <p className="subtitle">Review past scan predictions and analysis results.</p>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={fetchHistory} disabled={loading}>
                    <RefreshCw size={14} className={loading ? 'spinning' : ''} />
                    Refresh
                </button>
            </div>

            {error && (
                <div className="glass-card" style={{
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                    marginBottom: '1.5rem',
                    color: 'var(--severity-high)',
                }}>
                    {error}
                </div>
            )}

            {loading && (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <p>Loading history...</p>
                </div>
            )}

            {!loading && records.length === 0 && (
                <div className="glass-card history-empty">
                    <div className="history-empty-icon">
                        <Inbox size={28} />
                    </div>
                    <h3>No scans yet</h3>
                    <p>Upload an MRI scan to see results here.</p>
                </div>
            )}

            {!loading && records.length > 0 && (
                <div className="history-table-wrapper">
                    <table className="history-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>File</th>
                                <th>Tumor Type</th>
                                <th>Confidence</th>
                                <th>Severity</th>
                                <th>Date</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {records.map((record) => (
                                <tr key={record.id}>
                                    <td>{record.id}</td>
                                    <td>{record.original_filename || record.filename}</td>
                                    <td className="tumor-type">
                                        {record.tumor_type === 'notumor' ? 'No Tumor' : record.tumor_type}
                                    </td>
                                    <td>{(record.confidence * 100).toFixed(1)}%</td>
                                    <td>
                                        <span className={`severity-badge ${getSeverityClass(record.severity)}`}>
                                            {record.severity || 'N/A'}
                                        </span>
                                    </td>
                                    <td>{new Date(record.created_at).toLocaleDateString()}</td>
                                    <td>
                                        <button
                                            className="btn btn-danger btn-sm"
                                            onClick={() => handleDelete(record.id)}
                                            title="Delete record"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
