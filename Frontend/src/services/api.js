import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds for wallet analysis
});

// Health check
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('API is not responding');
  }
};

export const ping = async () => {
  const response = await api.get('/ping');
  return response.data;
};

// Wallet analysis
export const analyzeWallet = async (address, modelPath = '../outputs/gnn_model.pt') => {
  try {
    const response = await api.get(`/analyze/${address}`, {
      params: { model_path: modelPath }
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Analysis failed');
    }
    throw new Error('Network error. Please check if the backend is running.');
  }
};

// Fetch wallet transaction info
export const getWalletInfo = async (address) => {
  try {
    const response = await api.get(`/info/${address}`);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to fetch wallet info');
    }
    throw new Error('Network error. Please check if the backend is running.');
  }
};

// Get random wallet address
export const getRandomWallet = async () => {
  try {
    const response = await api.get('/random');
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to fetch random wallet');
    }
    throw new Error('Network error. Please check if the backend is running.');
  }
};

// Get feature importance
export const getFeatureImportance = async (address) => {
  try {
    const response = await api.get(`/feature-importance/${address}`);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Failed to fetch feature importance');
    }
    throw new Error('Network error. Please check if the backend is running.');
  }
};

export default {
  checkHealth,
  ping,
  analyzeWallet,
  getWalletInfo,
  getRandomWallet,
  getFeatureImportance,
};
