import { Activity } from 'lucide-react';
import WalletAnalysis from './pages/WalletAnalysis';
import './App.css';

function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <Activity size={28} />
            <h1>Bitcoin Wallet Analyzer</h1>
          </div>
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
