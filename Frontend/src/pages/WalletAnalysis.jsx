import { useState } from 'react';
import { Search, Loader, AlertCircle, CheckCircle, XCircle, TrendingUp, Network, Activity, ChevronDown, ChevronUp, ExternalLink, ArrowRightLeft, Wallet, ArrowDownLeft, ArrowUpRight } from 'lucide-react';
import { analyzeWallet, getWalletInfo } from '../services/api';

function WalletAnalysis() {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [walletInfo, setWalletInfo] = useState(null);
  const [loadingTxns, setLoadingTxns] = useState(false);
  const [error, setError] = useState('');
  const [expandedTx, setExpandedTx] = useState(null);
  const [showTransactions, setShowTransactions] = useState(false);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    
    if (!address.trim()) {
      setError('Please enter a Bitcoin wallet address');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setWalletInfo(null);
    setShowTransactions(false);

    try {
      // Fetch both analysis and transaction info in parallel
      const [analysisData, infoData] = await Promise.all([
        analyzeWallet(address.trim()),
        getWalletInfo(address.trim()).catch(() => null)
      ]);
      setResult(analysisData);
      setWalletInfo(infoData);
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

  const formatSatoshis = (sats) => {
    if (sats === null || sats === undefined) return 'N/A';
    if (Math.abs(sats) >= 100000000) {
      return `${(sats / 100000000).toFixed(8)} BTC`;
    }
    return `${sats.toLocaleString()} sats`;
  };

  const formatBTC = (btc) => {
    if (btc === null || btc === undefined) return 'N/A';
    return `${btc.toFixed(8)} BTC`;
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const truncateAddress = (addr, chars = 8) => {
    if (!addr) return '';
    if (addr.length <= chars * 2) return addr;
    return `${addr.slice(0, chars)}...${addr.slice(-chars)}`;
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

            {result.ghost_nodes !== undefined && result.ghost_nodes !== null && (
              <div className="stat-card">
                <div className="stat-icon orange">
                  <Network size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-label">Neighbors</div>
                  <div className="stat-value">{result.ghost_nodes}</div>
                </div>
              </div>
            )}

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

          {/* Wallet Balance Section */}
          {walletInfo && (
            <div className="balance-card">
              <div className="balance-header">
                <Wallet size={24} />
                <h3>Wallet Balance</h3>
              </div>
              <div className="balance-grid">
                <div className="balance-item main-balance">
                  <div className="balance-label">Current Balance</div>
                  <div className="balance-value btc">{formatBTC(walletInfo.balance_btc)}</div>
                  <div className="balance-subvalue">{walletInfo.balance_sats?.toLocaleString()} sats</div>
                </div>
                <div className="balance-item received">
                  <div className="balance-icon">
                    <ArrowDownLeft size={20} />
                  </div>
                  <div className="balance-content">
                    <div className="balance-label">Total Received</div>
                    <div className="balance-value">{formatSatoshis(walletInfo.total_received_sats)}</div>
                    <div className="balance-count">{walletInfo.funded_txo_count} inputs</div>
                  </div>
                </div>
                <div className="balance-item sent">
                  <div className="balance-icon">
                    <ArrowUpRight size={20} />
                  </div>
                  <div className="balance-content">
                    <div className="balance-label">Total Sent</div>
                    <div className="balance-value">{formatSatoshis(walletInfo.total_sent_sats)}</div>
                    <div className="balance-count">{walletInfo.spent_txo_count} outputs</div>
                  </div>
                </div>
              </div>
            </div>
          )}

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

          {/* Transactions Section */}
          {walletInfo && walletInfo.transactions && walletInfo.transactions.length > 0 && (
            <div className="transactions-card">
              <div 
                className="transactions-header"
                onClick={() => setShowTransactions(!showTransactions)}
              >
                <div className="transactions-title">
                  <ArrowRightLeft size={24} />
                  <h3>Transactions ({walletInfo.transaction_count})</h3>
                </div>
                <button className="toggle-btn">
                  {showTransactions ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </button>
              </div>

              {showTransactions && (
                <div className="transactions-list">
                  {walletInfo.transactions.slice(0, 20).map((tx, index) => (
                    <div key={tx.txid} className="transaction-item">
                      <div 
                        className="transaction-summary"
                        onClick={() => setExpandedTx(expandedTx === tx.txid ? null : tx.txid)}
                      >
                        <div className="tx-main-info">
                          <div className="tx-id">
                            <code>{truncateAddress(tx.txid, 12)}</code>
                            <a 
                              href={`https://mempool.space/tx/${tx.txid}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="external-link"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <ExternalLink size={14} />
                            </a>
                          </div>
                          <div className="tx-status">
                            {tx.status?.confirmed ? (
                              <span className="confirmed">
                                <CheckCircle size={14} />
                                Block {tx.status.block_height}
                              </span>
                            ) : (
                              <span className="pending">
                                <Loader size={14} />
                                Pending
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="tx-details-row">
                          <span className="tx-fee">Fee: {formatSatoshis(tx.fee)}</span>
                          <span className="tx-size">{tx.size} bytes</span>
                          {tx.status?.block_time && (
                            <span className="tx-time">{formatDate(tx.status.block_time)}</span>
                          )}
                        </div>
                        <button className="expand-btn">
                          {expandedTx === tx.txid ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>
                      </div>

                      {expandedTx === tx.txid && (
                        <div className="transaction-expanded">
                          <div className="tx-io-container">
                            <div className="tx-inputs">
                              <h4>Inputs ({tx.vin?.length || 0})</h4>
                              {tx.vin?.slice(0, 5).map((input, i) => (
                                <div key={i} className="tx-io-item">
                                  {input.is_coinbase ? (
                                    <span className="coinbase">Coinbase</span>
                                  ) : (
                                    <>
                                      <code>{truncateAddress(input.prevout?.scriptpubkey_address, 10)}</code>
                                      <span className="amount">{formatSatoshis(input.prevout?.value || 0)}</span>
                                    </>
                                  )}
                                </div>
                              ))}
                              {tx.vin?.length > 5 && (
                                <div className="more-items">+{tx.vin.length - 5} more inputs</div>
                              )}
                            </div>
                            <div className="tx-arrow">→</div>
                            <div className="tx-outputs">
                              <h4>Outputs ({tx.vout?.length || 0})</h4>
                              {tx.vout?.slice(0, 5).map((output, i) => (
                                <div key={i} className="tx-io-item">
                                  {output.scriptpubkey_type === 'op_return' ? (
                                    <span className="op-return">OP_RETURN</span>
                                  ) : (
                                    <>
                                      <code 
                                        className={output.scriptpubkey_address === result.wallet_address ? 'highlight' : ''}
                                      >
                                        {truncateAddress(output.scriptpubkey_address, 10)}
                                      </code>
                                      <span className="amount">{formatSatoshis(output.value)}</span>
                                    </>
                                  )}
                                </div>
                              ))}
                              {tx.vout?.length > 5 && (
                                <div className="more-items">+{tx.vout.length - 5} more outputs</div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  {walletInfo.transactions.length > 20 && (
                    <div className="more-transactions">
                      Showing 20 of {walletInfo.transaction_count} transactions.
                      <a 
                        href={`https://mempool.space/address/${result.wallet_address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        View all on Mempool.space <ExternalLink size={14} />
                      </a>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default WalletAnalysis;
