import { useState, useMemo, useEffect } from 'react';
import { Search, Loader, AlertCircle, CheckCircle, XCircle, TrendingUp, Network, Activity, ChevronDown, ChevronUp, ExternalLink, ArrowRightLeft, Wallet, ArrowDownLeft, ArrowUpRight, BarChart3, Download, Moon, Sun, History, FileJson } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { analyzeWallet, getWalletInfo, getRandomWallet } from '../services/api';

function WalletAnalysis() {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [walletInfo, setWalletInfo] = useState(null);
  const [loadingTxns, setLoadingTxns] = useState(false);
  const [error, setError] = useState('');
  const [expandedTx, setExpandedTx] = useState(null);
  const [showTransactions, setShowTransactions] = useState(false);
  const [visibleTxCount, setVisibleTxCount] = useState(20);
  const [searchHistory, setSearchHistory] = useState(() => {
    const saved = localStorage.getItem('searchHistory');
    return saved ? JSON.parse(saved) : [];
  });
  const [showHistory, setShowHistory] = useState(false);

  // Save search history
  useEffect(() => {
    localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  }, [searchHistory]);

  const addToHistory = (addr) => {
    const newEntry = {
      address: addr,
      timestamp: new Date().toISOString(),
      classification: result?.classification
    };
    setSearchHistory(prev => {
      const filtered = prev.filter(item => item.address !== addr);
      return [newEntry, ...filtered].slice(0, 10); // Keep last 10
    });
  };

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
    setVisibleTxCount(20);

    try {
      // Fetch both analysis and transaction info in parallel
      const [analysisData, infoData] = await Promise.all([
        analyzeWallet(address.trim()),
        getWalletInfo(address.trim()).catch(() => null)
      ]);
      setResult(analysisData);
      setWalletInfo(infoData);
      addToHistory(address.trim());
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

  // Prepare chart data - aggregate transaction volume by week
  const chartData = useMemo(() => {
    if (!walletInfo?.transactions?.length) return [];
    
    const volumeByPeriod = {};
    const walletAddr = result?.wallet_address;
    
    walletInfo.transactions.forEach(tx => {
      if (!tx.status?.block_time) return;
      
      const date = new Date(tx.status.block_time * 1000);
      // Use week-based aggregation for better granularity
      const startOfWeek = new Date(date);
      startOfWeek.setDate(date.getDate() - date.getDay());
      const weekKey = startOfWeek.toISOString().split('T')[0];
      
      // Calculate incoming (received) and outgoing (sent) for this wallet
      let incoming = 0;
      let outgoing = 0;
      
      // Check inputs (spending from our wallet = outgoing)
      tx.vin?.forEach(input => {
        if (input.prevout?.scriptpubkey_address === walletAddr) {
          outgoing += input.prevout.value || 0;
        }
      });
      
      // Check outputs (receiving to our wallet = incoming)
      tx.vout?.forEach(output => {
        if (output.scriptpubkey_address === walletAddr) {
          incoming += output.value || 0;
        }
      });
      
      if (!volumeByPeriod[weekKey]) {
        volumeByPeriod[weekKey] = { period: weekKey, incoming: 0, outgoing: 0, txCount: 0 };
      }
      volumeByPeriod[weekKey].incoming += incoming;
      volumeByPeriod[weekKey].outgoing += outgoing;
      volumeByPeriod[weekKey].txCount += 1;
    });
    
    return Object.values(volumeByPeriod)
      .sort((a, b) => a.period.localeCompare(b.period))
      .map(item => ({
        ...item,
        incomingBTC: item.incoming / 100000000,
        outgoingBTC: item.outgoing / 100000000,
        volumeBTC: (item.incoming + item.outgoing) / 100000000,
        label: new Date(item.period).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
      }));
  }, [walletInfo, result]);

  const generateRandomWallet = async () => {
    try {
      const data = await getRandomWallet();
      if (data.address) {
        setAddress(data.address);
      }
    } catch (err) {
      console.error('Error fetching random wallet:', err);
      setError('Failed to fetch random wallet. Please try again.');
    }
  };

  const exportAsJSON = () => {
    const exportData = {
      address: result.wallet_address,
      analysis: {
        classification: result.classification,
        risk_score: result.risk_score,
        confidence: result.confidence,
        nodes_count: result.nodes_count,
        edges_count: result.edges_count,
        ghost_nodes: result.ghost_nodes
      },
      balance: walletInfo ? {
        balance_btc: walletInfo.balance_btc,
        total_received_sats: walletInfo.total_received_sats,
        total_sent_sats: walletInfo.total_sent_sats
      } : null,
      exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wallet-analysis-${result.wallet_address.slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsPDF = async () => {
    const { jsPDF } = await import('jspdf');
    const doc = new jsPDF();
    
    // Title
    doc.setFontSize(20);
    doc.text('Bitcoin Wallet Risk Analysis', 20, 20);
    
    // Address
    doc.setFontSize(12);
    doc.text(`Address: ${result.wallet_address}`, 20, 35);
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 20, 42);
    
    // Classification
    doc.setFontSize(16);
    doc.text('Classification', 20, 55);
    doc.setFontSize(12);
    doc.text(`Result: ${result.classification}`, 20, 65);
    doc.text(`Risk Score: ${(result.risk_score * 100).toFixed(1)}%`, 20, 72);
    doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 20, 79);
    
    // Graph Statistics
    doc.setFontSize(16);
    doc.text('Graph Statistics', 20, 95);
    doc.setFontSize(12);
    doc.text(`Nodes: ${result.nodes_count}`, 20, 105);
    doc.text(`Edges: ${result.edges_count}`, 20, 112);
    doc.text(`Neighbors: ${result.ghost_nodes}`, 20, 119);
    
    // Balance Info
    if (walletInfo) {
      doc.setFontSize(16);
      doc.text('Balance Information', 20, 135);
      doc.setFontSize(12);
      doc.text(`Current Balance: ${walletInfo.balance_btc.toFixed(8)} BTC`, 20, 145);
      doc.text(`Total Received: ${(walletInfo.total_received_sats / 100000000).toFixed(8)} BTC`, 20, 152);
      doc.text(`Total Sent: ${(walletInfo.total_sent_sats / 100000000).toFixed(8)} BTC`, 20, 159);
      doc.text(`Transactions: ${walletInfo.transaction_count}`, 20, 166);
    }
    
    doc.save(`wallet-analysis-${result.wallet_address.slice(0, 10)}.pdf`);
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
            <p className="text-muted">Get a wallet from latest transactions:</p>
            <div className="address-examples">
              <button 
                type="button"
                className="example-btn random-btn"
                onClick={generateRandomWallet}
              >
                Random Wallet
              </button>
              <button 
                type="button"
                className="example-btn"
                onClick={() => setShowHistory(!showHistory)}
              >
                <History size={16} />
                History ({searchHistory.length})
              </button>
            </div>
          </div>

          {/* Search History Dropdown */}
          {showHistory && searchHistory.length > 0 && (
            <div className="history-dropdown">
              <h4>Recent Searches</h4>
              {searchHistory.map((item, idx) => (
                <div 
                  key={idx} 
                  className="history-item"
                  onClick={() => {
                    setAddress(item.address);
                    setShowHistory(false);
                  }}
                >
                  <div className="history-address">
                    <code>{item.address.slice(0, 20)}...{item.address.slice(-10)}</code>
                    {item.classification && (
                      <span className={`history-badge ${item.classification.toLowerCase()}`}>
                        {item.classification}
                      </span>
                    )}
                  </div>
                  <div className="history-time">
                    {new Date(item.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
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
            <div>
              <h2>Analysis Results</h2>
              <span className="status-badge success">
                <CheckCircle size={16} />
                {result.status}
              </span>
            </div>
            <div className="export-buttons">
              <button 
                className="btn btn-secondary"
                onClick={exportAsJSON}
                title="Export as JSON"
              >
                <FileJson size={18} />
                JSON
              </button>
              <button 
                className="btn btn-secondary"
                onClick={exportAsPDF}
                title="Export as PDF"
              >
                <Download size={18} />
                PDF
              </button>
            </div>
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

          {/* Transaction Volume Chart */}
          {chartData.length > 0 && (
            <div className="chart-card">
              <div className="chart-header">
                <BarChart3 size={24} />
                <h3>Transaction Volume Over Time</h3>
              </div>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis 
                      dataKey="label" 
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                    />
                    <YAxis 
                      tick={{ fontSize: 12, fill: '#6b7280' }}
                      axisLine={{ stroke: '#d1d5db' }}
                      tickFormatter={(value) => value.toFixed(4)}
                      label={{ value: 'BTC', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#6b7280' } }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                      }}
                      formatter={(value, name) => {
                        const labels = { incomingBTC: 'Received', outgoingBTC: 'Sent' };
                        return [`${value.toFixed(8)} BTC`, labels[name] || name];
                      }}
                      labelFormatter={(label) => `Week of ${label}`}
                    />
                    <Legend 
                      formatter={(value) => value === 'incomingBTC' ? 'Received' : 'Sent'}
                      wrapperStyle={{ paddingTop: '10px' }}
                    />
                    <Bar 
                      dataKey="incomingBTC" 
                      fill="#22c55e" 
                      radius={[4, 4, 0, 0]}
                      name="incomingBTC"
                    />
                    <Bar 
                      dataKey="outgoingBTC" 
                      fill="#ef4444" 
                      radius={[4, 4, 0, 0]}
                      name="outgoingBTC"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-summary">
                <div className="chart-stat received">
                  <span className="chart-stat-label">Total Received</span>
                  <span className="chart-stat-value">{chartData.reduce((sum, d) => sum + d.incomingBTC, 0).toFixed(6)} BTC</span>
                </div>
                <div className="chart-stat">
                  <span className="chart-stat-label">Transactions</span>
                  <span className="chart-stat-value">{chartData.reduce((sum, d) => sum + d.txCount, 0)}</span>
                </div>
                <div className="chart-stat sent">
                  <span className="chart-stat-label">Total Sent</span>
                  <span className="chart-stat-value">{chartData.reduce((sum, d) => sum + d.outgoingBTC, 0).toFixed(6)} BTC</span>
                </div>
              </div>
            </div>
          )}

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
                  {walletInfo.transactions.slice(0, visibleTxCount).map((tx, index) => (
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
                  {visibleTxCount < walletInfo.transactions.length && (
                    <div className="load-more-container">
                      <button 
                        className="btn btn-secondary load-more-btn"
                        onClick={() => setVisibleTxCount(prev => prev + 20)}
                      >
                        Load More Transactions
                      </button>
                      <span className="load-more-info">
                        Showing {visibleTxCount} of {walletInfo.transaction_count}
                      </span>
                    </div>
                  )}
                  {visibleTxCount >= walletInfo.transactions.length && walletInfo.transactions.length > 20 && (
                    <div className="more-transactions">
                      All {walletInfo.transaction_count} transactions loaded.
                      <a 
                        href={`https://mempool.space/address/${result.wallet_address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        View on Mempool.space <ExternalLink size={14} />
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
