import axios from 'axios';

const runtimeEnv = typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env : {};

const api = axios.create({
  baseURL: runtimeEnv.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 10000,
});

export const checkHealth = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export const detectImage = async (file, algorithm = 'classic') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('algorithm', algorithm);
  return api.post('/api/detect/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const detectVideo = async (file, algorithm = 'classic') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('algorithm', algorithm);
  return api.post('/api/detect/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 600000,
  });
};

export const getAnalytics = async () => {
  const response = await api.get('/api/history/analytics');
  return response.data;
};

export const getDetectionHistory = async (page = 1, pageSize = 20, filters = {}) => {
  const response = await api.get('/api/history/detections', {
    params: {
      page,
      page_size: pageSize,
      ...filters,
    },
  });
  return response.data;
};

export const getDetectionDetail = async (detectionId) => {
  const response = await api.get(`/api/history/detections/${detectionId}`);
  return response.data;
};

export default api;
