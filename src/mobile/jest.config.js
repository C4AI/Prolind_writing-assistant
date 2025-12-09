module.exports = {
  preset: 'react-native',
  setupFiles: ['./jestSetup.js'], 
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|@react-navigation|@expo/vector-icons|react-native-vector-icons|react-native-gesture-handler)/)',
  ],
  moduleNameMapper: {
    '\\.png$': '<rootDir>/__mocks__/fileMock.js',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.svg': '<rootDir>/__mocks__/svgMock.js'// Adicione este mapeamento para ignorar arquivos .png
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
  },
};
