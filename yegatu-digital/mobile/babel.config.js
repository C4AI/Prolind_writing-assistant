module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['module:@react-native/babel-preset'],
    plugins: [
      'nativewind/babel',
      ["@babel/plugin-transform-class-properties", { "loose": true }],
      ["@babel/plugin-transform-private-methods", { "loose": true }],
      ["@babel/plugin-transform-private-property-in-object", { "loose": true }],
      '@babel/plugin-transform-flow-strip-types',
      'react-native-reanimated/plugin',
      ['module:react-native-dotenv', {
        moduleName: '@env',
        path: './.env',
      }],
    ],
  };
};