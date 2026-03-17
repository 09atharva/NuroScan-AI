import { Link } from 'react-router-dom';
import { Brain, Upload, Activity, Database, Shield, Zap } from 'lucide-react';

export default function Home() {
    return (
        <div className="home-page">
            {/* Hero */}
            <section className="hero">
                <div className="hero-badge">
                    <Zap size={14} />
                    Powered by SpineNet AI
                </div>
                <h1>
                    Brain Tumor<br />
                    <span className="gradient-text">Classification System</span>
                </h1>
                <p>
                    Upload brain MRI scans for instant AI-powered tumor type identification
                    and severity assessment. Built with SpineNet's multi-scale architecture
                    for accurate, reliable analysis.
                </p>
                <div className="hero-cta">
                    <Link to="/upload" className="btn btn-primary">
                        <Upload size={18} />
                        Upload Scan
                    </Link>
                    <Link to="/history" className="btn btn-secondary">
                        <Activity size={18} />
                        View History
                    </Link>
                </div>
            </section>

            {/* Stats */}
            <div className="container">
                <div className="stats-grid">
                    <div className="glass-card stat-card">
                        <div className="stat-value">7,023</div>
                        <div className="stat-label">Training Images</div>
                    </div>
                    <div className="glass-card stat-card">
                        <div className="stat-value">4</div>
                        <div className="stat-label">Tumor Classes</div>
                    </div>
                    <div className="glass-card stat-card">
                        <div className="stat-value">SpineNet</div>
                        <div className="stat-label">AI Backbone</div>
                    </div>
                    <div className="glass-card stat-card">
                        <div className="stat-value">&gt;91.5%</div>
                        <div className="stat-label">Target Accuracy</div>
                    </div>
                </div>

                {/* Feature Cards */}
                <div className="stats-grid" style={{ marginTop: '2rem' }}>
                    <div className="glass-card" style={{ padding: '2rem' }}>
                        <Brain size={32} style={{ color: 'var(--accent-primary)', marginBottom: '1rem' }} />
                        <h3 style={{ marginBottom: '0.5rem' }}>Tumor Classification</h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            Identifies glioma, meningioma, pituitary tumors, or confirms no tumor
                            present in MRI scans.
                        </p>
                    </div>
                    <div className="glass-card" style={{ padding: '2rem' }}>
                        <Shield size={32} style={{ color: 'var(--accent-secondary)', marginBottom: '1rem' }} />
                        <h3 style={{ marginBottom: '0.5rem' }}>Severity Assessment</h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            Provides risk severity levels (Low / Moderate / High) based on
                            prediction confidence analysis.
                        </p>
                    </div>
                    <div className="glass-card" style={{ padding: '2rem' }}>
                        <Database size={32} style={{ color: 'var(--severity-low)', marginBottom: '1rem' }} />
                        <h3 style={{ marginBottom: '0.5rem' }}>Scan History</h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            All predictions are stored securely for future reference, review,
                            and comparative analysis.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
