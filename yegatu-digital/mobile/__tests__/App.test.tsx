/**
 * @format
 */

import 'react-native';
import React from 'react';
import App from '../src/screens/App';
import { render } from '@testing-library/react-native';

jest.mock('@expo/vector-icons', () => ({
  Feather: 'Feather',
}));

jest.mock('@react-native-clipboard/clipboard', () => ({
  setString: jest.fn(),
  getString: jest.fn(),
}));


describe('<App />', () => {
  it('renders correctly', () => {
    const { getByText } = render(<App />);
  expect(getByText("Pinimasa "+'\n'+" yẽgatu rupi")).toBeTruthy();
  });
});

// Mock do NetInfo
jest.mock('@react-native-community/netinfo', () => ({
  addEventListener: jest.fn(),
  fetch: jest.fn(() => Promise.resolve({ isConnected: true })),
  useNetInfo: jest.fn(() => ({ isConnected: true })),
}));

describe('<App />', () => {
  it('renders correctly', () => {
    const { getByText } = render(<App />);
    // Adicione aqui suas asserções de teste
  });
});