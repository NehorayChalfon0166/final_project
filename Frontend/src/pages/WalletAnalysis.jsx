import { useState } from 'react';
import { Search, Loader, AlertCircle, CheckCircle, XCircle, TrendingUp, Network, Activity } from 'lucide-react';
import { analyzeWallet } from '../services/api';

function WalletAnalysis() {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async (e) => {
    e.preventDefault();
    
    if (!address.trim()) {
      setError('Please enter a Bitcoin wallet address');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await analyzeWallet(address.trim());
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze wallet');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (classification) => {
    if (!classification) return 'gray';
    return classification.toLowerCase() === 'criminal' ? 'red' : 'green';
  };

  const getRiskLevel = (riskScore) => {
    if (riskScore >= 0.7) return 'High';
    if (riskScore >= 0.4) return 'Medium';
    return 'Low';
  };

  const generateRandomWallet = () => {
    // Real Bitcoin addresses with confirmed activity on mempool.space
    const realWallets = [
      '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',  // Satoshi Nakamoto - Genesis block
      'bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97',  // Binance cold wallet
      '3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r',  // Bitfinex
      'bc1qa5wkgaew2dkv56kfvj49j0av5nml45x9ek9hz6',  // Active SegWit wallet
      '1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s',  // Bittrex
      '3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64',  // Huobi
      'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h',  // Active modern wallet
      '12cbQLTFMXRnSzktFkuoG3eHJjPyoPLgy7',  // Early mining pool
      '1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g',  // Kraken
      'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh'   // Active SegWit address
    ];
    const randomAddress = realWallets[Math.floor(Math.random() * realWallets.length)];
    setAddress(randomAddress);
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Bitcoin Wallet Risk Analysis</h1>
        <p>Analyze Bitcoin wallets using Graph Neural Networks</p>
      </div>

      <div className="analysis-form-card">
        <form onSubmit={handleAnalyze} className="analysis-form">
          <div className="form-group">
            <label htmlFor="wallet-address">Bitcoin Wallet Address</label>
            <div className="input-with-button">
              <input
                id="wallet-address"
                type="text"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="Enter Bitcoin wallet address"
                className="form-input"
                disabled={loading}
              />
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader className="spin" size={20} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search size={20} />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </div>
          
          <div className="example-addresses">
            <p className="text-muted">Example address:</p>
            <div className="address-examples">
              <button 
                type="button"
                className="example-btn"
                onClick={() => setAddress('1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa')}
              >
                Bitcoin Genesis Block
              </button>
              <button 
                type="button"
                className="example-btn random-btn"
                onClick={generateRandomWallet}
              >
                Random Wallet
              </button>
            </div>
          </div>
        </form>

        {error && (
          <div className="alert alert-error">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {result && (
        <div className="results-container">
          <div className="result-header">
            <h2>Analysis Results</h2>
            <span className="status-badge success">
              <CheckCircle size={16} />
              {result.status}
            </span>
          </div>

          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon blue">
                <Network size={24} />
              </div>
              <div className="stat-content">
                <div className="stat-label">Nodes</div>
                <div className="stat-value">{result.nodes_count}</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon purple">
                <Activity size={24} />
              </div>
              <div className="stat-content">
                <div className="stat-label">Edges</div>
                <div className="stat-value">{result.edges_count}</div>
              </div>
            </div>

            {result.risk_score !== null && (
              <div className="stat-card">
                <div className={`stat-icon ${getRiskColor(result.classification)}`}>
                  <TrendingUp size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-label">Risk Score</div>
                  <div className="stat-value">
                    {(result.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}

            {result.confidence !== null && (
              <div className="stat-card">
                <div className="stat-icon green">
                  <CheckCircle size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-label">Confidence</div>
                  <div className="stat-value">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>

          {result.classification && (
            <div className={`classification-card ${getRiskColor(result.classification)}`}>
              <div className="classification-header">
                {result.classification.toLowerCase() === 'criminal' ? (
                  <XCircle size={32} />
                ) : (
                  <CheckCircle size={32} />
                )}
                <h3>Classification: {result.classification}</h3>
              </div>
              <p className="classification-description">
                {result.classification.toLowerCase() === 'criminal' 
                  ? 'This wallet has been classified as potentially involved in criminal activity.'
                  : 'This wallet appears to be legitimate with normal transaction patterns.'
                }
              </p>
              {result.risk_score !== null && (
                <div className="risk-level">
                  <span>Risk Level: </span>
                  <strong>{getRiskLevel(result.risk_score)}</strong>
                </div>
              )}
            </div>
          )}

          <div className="details-card">
            <h3>Wallet Details</h3>
            <div className="detail-row">
              <span className="detail-label">Address:</span>
              <code className="detail-value">{result.wallet_address}</code>
            </div>
            <div className="detail-row">
              <span className="detail-label">Graph Structure:</span>
              <div className="detail-value">
                {result.graph_data ? (
                  <>
                    <div>Node features: {result.graph_data.x_shape?.join(' × ')}</div>
                    <div>Edge index: {result.graph_data.edge_index_shape?.join(' × ')}</div>
                    <div>Edge attributes: {result.graph_data.edge_attr_shape?.join(' × ')}</div>
                  </>
                ) : (
                  <div>No graph data available</div>
                )}
              </div>
            </div>
            {result.message && (
              <div className="detail-row">
                <span className="detail-label">Message:</span>
                <span className="detail-value">{result.message}</span>
              </div>
            )}
            {result.inference_error && (
              <div className="alert alert-warning">
                <AlertCircle size={16} />
                <span>{result.inference_error}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default WalletAnalysis;
