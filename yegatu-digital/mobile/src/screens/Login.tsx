import React, { useEffect, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  Alert,
  KeyboardAvoidingView,
  ScrollView,
  Platform,
  Button,
  TouchableOpacity,
  Image,
  Pressable,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import CustomTextInput from "../components/CustomTextInput";
import CustomTextInputWithImage from "../components/CustomTextInputWithImage";
import LoginButton from "../components/LoginButton";
import { StackNavigationProp } from "@react-navigation/stack";
import { RootStackParamList } from "../types/navigation";
import axios, { isAxiosError } from "axios";
import { useAuth } from "../utils/AuthContext";
import Config from "react-native-config";
import WebView from "react-native-webview";
import RNFS from "react-native-fs";
import AsyncStorage from "@react-native-async-storage/async-storage";
import FeedbackModal from "../components/FeedbackModal";

type MainScreenProps = StackNavigationProp<RootStackParamList, "MainPage">;

export default function LoginScreen() {
  const navigation = useNavigation<MainScreenProps>();
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [app_id, setApp_id] = useState(0);
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const { token, setToken } = useAuth();
  const { tokenEn, setTokenEn } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [openWebView, setOpenWebView] = useState(false);
  const apiUrl =
  //  "https://assistente-escrita-linguas-indigenas-backend.y6dbcklf96p.us-south.codeengine.appdomain.cloud";
  //  "http://10.0.2.2:3000";
    "http://104.154.155.83";

  const [version, setVersion] = useState<string>("");
  const [feedbackModal, setFeedbackModal] = useState({
    visible: false,
    title: "",
    message: "",
    type: "info" as "error" | "success" | "info" | "warning",
    isTempPopUp: false,
  });

  const resetCredentials = () => {
    setUsername("");
    setPassword("");
  };

  useEffect(() => {
    const readVersionFile = async () => {
      try {
        const content = await RNFS.readFileAssets("version.txt", "utf8");
        setVersion(content);
      } catch (err) {}
    };

    readVersionFile();
  }, []);

  useEffect(() => {
    const unsubscribe = navigation.addListener("focus", () => {
      const checkSavedCredentials = async () => {
        const storedUsername = await AsyncStorage.getItem("username");
        const storedPassword = await AsyncStorage.getItem("password");

        if (storedUsername && storedPassword) {
          setUsername(storedUsername);
          setPassword(storedPassword);
          handleLogin(storedUsername, storedPassword);
        }
      };

      checkSavedCredentials();
    });

    return unsubscribe;
  }, [navigation]);

  const handleLogin = async (
    usernameFromProps?: string,
    passwordFromProps?: string
  ) => {
    try {
      setIsLoading(true);

      let response;
      if (!usernameFromProps || !passwordFromProps) {
        response = await axios.post(
          apiUrl + "/auth",
          {
            username,
            password,
          },
          {
            timeout: 20000,
          }
        );
      } else {
        response = await axios.post(
          apiUrl + "/auth",
          {
            username: usernameFromProps,
            password: passwordFromProps,
          },
          {
            timeout: 20000,
          }
        );
      }

      console.log("Login response:", response.data);
      await AsyncStorage.setItem(
        "username",
        username || usernameFromProps || ""
      );
      await AsyncStorage.setItem(
        "password",
        password || passwordFromProps || ""
      );

      axios.defaults.headers.common["Authorization"] = response.data.token;
      setToken(response.data.token);

      resetCredentials();
      navigation.navigate("MainPage", { username });
    } catch (error: any) {
      if (error.message === "Network Error" || error.code === "ECONNABORTED") {
        console.log("ESSE ERRO");
        Alert.alert("Erro de Conexão", "Verifique sua conexão com a internet.");
        console.log("Error:", error);
      } else {
        console.log("FOI");
        console.log(error.message);
        if (isAxiosError(error)) {
          setFeedbackModal({
            isTempPopUp: false,
            message: error.response?.data.message,
            title: "Falha no login",
            type: "error",
            visible: true,
          });
          return;
        }
        console.log("Error:", error);
        Alert.alert("Usuário ou senha inválidos");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setIsPasswordVisible(!isPasswordVisible);
  };

  return (
    <SafeAreaView className="flex-1 bg-white">
      <FeedbackModal
        visible={feedbackModal.visible}
        title={feedbackModal.title}
        message={feedbackModal.message}
        type={feedbackModal.type}
        isTempPopUp={feedbackModal.isTempPopUp}
        font={"font"}
        onClose={() =>
          setFeedbackModal((prev) => ({ ...prev, visible: false }))
        }
      />
      {openWebView ? (
        <>
          <TouchableOpacity
            className={`flex-row items-center justify-between w-full bg-purple-900 p-4 rounded`}
            onPress={() => setOpenWebView(false)}
            accessible={true}
            accessibilityRole="button"
          >
            <Image
              source={require("../assets/white-arrow.png")}
              className="w-6 h-2 tint-white rotate-180 "
            />
            <Text className="text-white text-lg text-left w-full px-5">
              Voltar
            </Text>
          </TouchableOpacity>
          <WebView
            source={{
              uri: "https://docs.google.com/forms/d/e/1FAIpQLScJj8Z8PjHlFKPIAUqA_pleEGIuA8kVsCuwsTM30I9TQFgMyA/viewform?pli=1",
            }}
          ></WebView>
        </>
      ) : (
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          className="flex-1"
        >
          <ScrollView
            contentContainerStyle={{
              flexGrow: 1,
              justifyContent: "center",
              padding: 24,
            }}
            keyboardShouldPersistTaps="handled"
          >
            <View className="flex-1 justify-center">
              <Text
                className="text-3xl font-bold mb-4 text-black text-center p-6 mt-auto"
                accessibilityLabel="Pinimasa yẽgatu rupi"
                accessibilityRole="header"
              >
                Pinimasa {"\n"} yẽgatu rupi
              </Text>
              <CustomTextInput
                placeholder="Digite seu usuário"
                value={username}
                onChangeText={setUsername}
                accessibilityLabel="Campo para inserir seu usuário"
              />
              <CustomTextInputWithImage
                placeholder="Digite sua senha"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!isPasswordVisible}
                onToggleVisibility={togglePasswordVisibility}
                iconSource={
                  !isPasswordVisible
                    ? require("../assets/hide-password.png")
                    : require("../assets/show-password.png")
                }
                accessibilityLabel="Campo para inserir sua senha"
              />
              <LoginButton
                additionalStyle="mt-8"
                title={isLoading ? "Carregando..." : "Continuar"}
                onPress={handleLogin}
                iconSource={require("../assets/white-arrow.png")}
                accessibilityLabel="Botão para concluir login"
                disabled={isLoading}
                color={isLoading ? "bg-gray-400" : "bg-blue-500"}
              />
              <Pressable className="mt-4" onPress={() => setOpenWebView(true)}>
                <Text className="text-blue-600 text-center  underline font-medium text-base active:opacity-70 ">
                  Solicitar cadastro
                </Text>
              </Pressable>
              <View className="items-center justify-center mb-6 w-full mt-auto">
                <Text className="font-medium text-center text-xs text-gray-500 mb-2">
                  Copyright 2025, Centro de Inteligência Artificial (C4AI),
                  Universidade de São Paulo.
                </Text>
                <Image
                  source={require("../assets/logo-c4ai.png")}
                  style={{
                    width: 100,
                    height: 40,
                    resizeMode: "contain",
                  }}
                />
              </View>
            </View>
            {version && (
              <Text className="font-medium text-center text-xs text-gray-500 absolute top-2 right-2">
                Versão {version}
              </Text>
            )}
          </ScrollView>
        </KeyboardAvoidingView>
      )}
    </SafeAreaView>
  );
}
