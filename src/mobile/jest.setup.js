// Exemplo de configuração de mock global
global.fetch = require("jest-fetch-mock");

try {
  require("react-native-gesture-handler/jestSetup");
} catch (error) {
  console.warn("react-native-gesture-handler/jestSetup not found. Skipping.");
}

jest.mock("react-native-reanimated", () => {
  const Reanimated = require("react-native-reanimated/mock");
  Reanimated.default.call = () => {};
  return Reanimated;
});

jest.mock("react-native/Libraries/Animated/NativeAnimatedHelper");

// Mock para o Modal
jest.mock("react-native", () => {
  const RN = jest.requireActual("react-native");
  RN.Modal = ({ children }) => children;
  return RN;
});
