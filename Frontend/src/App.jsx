import { Activity, Moon, Sun } from 'lucide-react';
import { useState, useEffect } from 'react';
import WalletAnalysis from './pages/WalletAnalysis';
import './App.css';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <Activity size={28} />
            <h1>Bitcoin Wallet Analyzer</h1>
          </div>
          <button 
            className="theme-toggle-nav"
            onClick={() => setDarkMode(!darkMode)}
            title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </nav>
      <main className="main-content">
        <WalletAnalysis />
      </main>
      <footer className="footer">
        <p>© 2026 Bitcoin Wallet Analyzer | Powered by GNN Technology</p>
      </footer>
    </div>
  );
}

export default App;
