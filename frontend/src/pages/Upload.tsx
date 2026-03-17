import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload as UploadIcon, X, FileImage, AlertCircle } from 'lucide-react';
import { uploadScan, type PredictionResult } from '../api/client';

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

    const handleFile = useCallback((selectedFile: File) => {
        if (!selectedFile.type.startsWith('image/')) {
            setError('Please upload an image file (JPEG, PNG).');
            return;
        }
        setError(null);
        setFile(selectedFile);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(selectedFile);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const dropped = e.dataTransfer.files[0];
        if (dropped) handleFile(dropped);
    }, [handleFile]);

    const handleSubmit = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);

        try {
            const result: PredictionResult = await uploadScan(file);
            // Navigate to results with the prediction data
            navigate('/results', { state: { result } });
        } catch (err: any) {
            setError(err.message || 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const clearFile = () => {
        setFile(null);
        setPreview(null);
        setError(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    return (
        <div className="upload-page">
            <h1>Upload MRI Scan</h1>
            <p className="subtitle">
                Upload a brain MRI scan image for AI-powered tumor classification and severity assessment.
            </p>

            {/* Error Message */}
            {error && (
                <div className="glass-card" style={{
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                    marginBottom: '1.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    color: 'var(--severity-high)',
                }}>
                    <AlertCircle size={20} />
                    <span>{error}</span>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div className="glass-card loading-overlay">
                    <div className="spinner" />
                    <h3>Analyzing MRI Scan...</h3>
                    <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                        Running SpineNet inference — this may take a few seconds.
                    </p>
                </div>
            )}

            {/* Dropzone */}
            {!loading && !file && (
                <div
                    className={`dropzone ${dragOver ? 'drag-over' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <div className="dropzone-content">
                        <div className="dropzone-icon">
                            <UploadIcon size={28} />
                        </div>
                        <h3>Drop your MRI scan here</h3>
                        <p>or click to browse — supports JPEG, PNG</p>
                    </div>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                            const f = e.target.files?.[0];
                            if (f) handleFile(f);
                        }}
                        style={{ display: 'none' }}
                    />
                </div>
            )}

            {/* File Preview */}
            {!loading && file && preview && (
                <>
                    <div className="glass-card file-preview">
                        <img src={preview} alt="MRI Preview" />
                        <div className="file-info">
                            <h4>{file.name}</h4>
                            <p>{formatSize(file.size)} · {file.type}</p>
                        </div>
                    </div>
                    <div className="upload-actions">
                        <button className="btn btn-primary" onClick={handleSubmit}>
                            <FileImage size={18} />
                            Analyze Scan
                        </button>
                        <button className="btn btn-secondary" onClick={clearFile}>
                            <X size={18} />
                            Remove
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
