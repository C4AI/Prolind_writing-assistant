/** @type {import('ts-jest').JestConfigWithTsJest} **/
export default {
  clearMocks: true,
  coverageProvider: "v8",
  moduleFileExtensions: ["js", "jsx", "ts", "tsx", "json", "node"],
  roots: ["<rootDir>/src"],
  testMatch: ["**/__tests__/**/*.[jt]s?(x)", "**/?(*.)+(spec|test).[tj]s?(x)"],
  transform: {
    "\\.[jt]sx?$": ["ts-jest", { useESM: true }],
  },
  globals: {
    "ts-jest": {
      useESM: true,
    },
  },
  moduleNameMapper: {
    "^(\\.{1,2}/.*)\\.js$": "$1",
  },
  extensionsToTreatAsEsm: [".ts"],
  modulePathIgnorePatterns: ["__mocks__"],
  setupFiles: ["./src/setup_tests.ts"],
};
