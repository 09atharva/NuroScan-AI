import { useLocation, Link, Navigate } from 'react-router-dom';
import { ArrowLeft, Upload, AlertTriangle } from 'lucide-react';
import { type PredictionResult, getImageUrl } from '../api/client';

export default function Results() {
    const location = useLocation();
    const result = location.state?.result as PredictionResult | undefined;

    if (!result) {
        return <Navigate to="/upload" replace />;
    }

    const severityClass = result.severity.toLowerCase() as 'high' | 'moderate' | 'low' | 'none';
    const confidencePercent = (result.confidence * 100).toFixed(1);
    const maxClass = result.tumor_type;

    return (
        <div className="results-page">
            <h1>Analysis Results</h1>

            <div className="result-layout">
                {/* Scan Image */}
                <div className="glass-card scan-image-card">
                    <img
                        src={getImageUrl(result.image_url)}
                        alt="Uploaded MRI scan"
                    />
                    <div className="image-label">
                        Uploaded scan · {new Date(result.created_at).toLocaleString()}
                    </div>
                </div>

                {/* Results */}
                <div className="result-details">
                    {/* Tumor Type + Severity */}
                    <div className="glass-card">
                        <div className="result-header">
                            <div className={`result-type-icon ${severityClass}`}>
                                <AlertTriangle size={24} />
                            </div>
                            <div>
                                <div className="result-type-name">{result.tumor_type === 'notumor' ? 'No Tumor' : result.tumor_type}</div>
                                <div className="result-severity">
                                    <span className={`severity-badge ${severityClass}`}>
                                        {result.severity} Severity
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Confidence */}
                    <div className="glass-card confidence-section">
                        <h3>Confidence</h3>
                        <div className="confidence-bar">
                            <div
                                className={`confidence-fill ${severityClass}`}
                                style={{ width: `${confidencePercent}%` }}
                            />
                        </div>
                        <div className="confidence-value">{confidencePercent}%</div>
                    </div>

                    {/* All Scores */}
                    <div className="glass-card all-scores">
                        <h3>Class Probabilities</h3>
                        {Object.entries(result.all_scores)
                            .sort(([, a], [, b]) => b - a)
                            .map(([cls, score]) => (
                                <div className="score-row" key={cls}>
                                    <span className="score-label">
                                        {cls === 'notumor' ? 'No Tumor' : cls}
                                    </span>
                                    <div className="score-bar-bg">
                                        <div
                                            className={`score-bar-fill ${cls === maxClass ? 'active' : ''}`}
                                            style={{ width: `${(score * 100).toFixed(1)}%` }}
                                        />
                                    </div>
                                    <span className="score-value">
                                        {(score * 100).toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                    </div>

                    {/* Actions */}
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <Link to="/upload" className="btn btn-primary">
                            <Upload size={18} />
                            New Scan
                        </Link>
                        <Link to="/history" className="btn btn-secondary">
                            <ArrowLeft size={18} />
                            View History
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}
