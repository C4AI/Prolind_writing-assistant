import React from "react";
import { createStackNavigator } from "@react-navigation/stack";
import { NavigationContainer } from "@react-navigation/native";
import LoginScreen from "./Login";
import MainPage from "./MainPage";
import { AuthProvider } from "../utils/AuthContext";
import FeedbackPage from "./FeedbackPage";
import { Text } from "react-native";
import { FeedbackProvider } from "../contexts/FeedbackContext";

const Stack = createStackNavigator();

export default function App() {
  return (
    <AuthProvider>
      <FeedbackProvider>
        <NavigationContainer>
          <Stack.Navigator
            initialRouteName="Login"
            screenOptions={{
              headerShown: false,
            }}
          >
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="MainPage" component={MainPage} />
            <Stack.Screen name="FeedbackPage" component={FeedbackPage} />
          </Stack.Navigator>
        </NavigationContainer>
      </FeedbackProvider>
    </AuthProvider>
  );
}
