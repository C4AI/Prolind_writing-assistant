export default {
    addEventListener: jest.fn(),
    fetch: jest.fn(() => Promise.resolve({ isConnected: true })),
    useNetInfo: jest.fn(() => ({ isConnected: true })),
  };