import test from 'node:test';
import assert from 'node:assert/strict';
import api, { getDetectionHistory } from './api.js';

test('getDetectionHistory requests the paginated detections endpoint', async () => {
  const originalGet = api.get;
  const calls = [];

  api.get = async (url, config = {}) => {
    calls.push({ url, config });
    return { data: { items: [] } };
  };

  try {
    const result = await getDetectionHistory(2, 10);

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/history/detections');
    assert.deepEqual(calls[0].config, { params: { page: 2, page_size: 10 } });
    assert.deepEqual(result, { items: [] });
  } finally {
    api.get = originalGet;
  }
});
