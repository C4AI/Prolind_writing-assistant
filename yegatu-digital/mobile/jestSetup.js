jest.mock('react-native-gesture-handler', () => {
    const View = require('react-native').View;
    return {
      Swipeable: View,
      DrawerLayout: View,
      PanGestureHandler: View, // Adicione essa linha para mockar PanGestureHandler
      State: {},
      GestureHandlerRootView: ({ children }) => children,
      Directions: {},
    };
  });
  