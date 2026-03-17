import { Link, useLocation } from 'react-router-dom';
import { Brain, Upload, History, Home } from 'lucide-react';

export default function Navbar() {
    const location = useLocation();

    const isActive = (path: string) => location.pathname === path ? 'active' : '';

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <Link to="/" className="navbar-brand">
                    <div className="brand-icon">
                        <Brain size={20} />
                    </div>
                    NeuroScan AI
                </Link>
                <ul className="navbar-links">
                    <li>
                        <Link to="/" className={isActive('/')}>
                            <Home size={16} />
                            <span className="link-text">Home</span>
                        </Link>
                    </li>
                    <li>
                        <Link to="/upload" className={isActive('/upload')}>
                            <Upload size={16} />
                            <span className="link-text">Upload</span>
                        </Link>
                    </li>
                    <li>
                        <Link to="/history" className={isActive('/history')}>
                            <History size={16} />
                            <span className="link-text">History</span>
                        </Link>
                    </li>
                </ul>
            </div>
        </nav>
    );
}
