import React, { useEffect } from "react";
import { Image } from "react-native";
import {
  SafeAreaView,
  StatusBar,
  View,
  Text,
  TouchableOpacity,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import { useState } from "react";
import thumbsUpOutline from "../../assets/thumbs-up.png";
import thumbsUpFilled from "../../assets/thumbs-up-pressed.png";
import thumbsDownOutline from "../../assets/thumbs-down.png";
import thumbsDownFilled from "../../assets/thumbs-down-pressed.png";
import { useFeedback } from "../../contexts/FeedbackContext";

// Imports necessários para a API e autenticação
import { useAuth } from "../../utils/AuthContext";
import axios, { isAxiosError } from "axios";
import EnviandoFeedbackButton from "../../components/EnviandoFeedbackButton";
import { suggestionData } from "../../components/SuggestionBox";

const FeedbackPage: React.FC = () => {
  const navigation = useNavigation();
  const [isLoading, setIsLoading] = useState(false);

  // API - Hook para autenticação
  const { token } = useAuth();

  const headerS = {
    Authorization: token,
    "Content-Type": "application/json",
    accept: "application/json",
  };

  // Timestamp
  const date = new Date();
  const timestamp = date.getTime();
  const apiUrl =
  //  "https://assistente-escrita-linguas-indigenas-backend.y6dbcklf96p.us-south.codeengine.appdomain.cloud";
  //  "http://10.0.2.2:3000";
    "http://104.154.155.83";


  // Estados para feedback
  const [generalFeedback, setGeneralFeedback] = useState(null);
  const [translationFeedback, setTranslationFeedback] = useState(null);
  const [correctionFeedback, setCorrectionFeedback] = useState(null);
  const [suggestionFeedback, setSuggestionFeedback] = useState(null);

  // Informações recebidas de MainPage
  const {
    translationLogs,
    suggestionLogs,
    correctionLogs,
    topLanguage,
    topOrthography,
    disableDic,
    disableNext,
    disableWordMeaning,
    dataCollection: state_data_collection,
    username,
    clearFeedbackData,
  } = useFeedback();

  console.log("FEEDBACK CONTEXT:", {
    translationLogs,
    suggestionLogs,
    correctionLogs,
    topLanguage,
    topOrthography,
    disableDic,
    disableNext,
    disableWordMeaning,
    dataCollection: state_data_collection,
    username,
    clearFeedbackData,
  });

  // Handle para voltar à MainPage
  const handleNavigateToMainPage = () => {
    navigation.navigate("MainPage", {
      resetFeedback: true,
      username: username,
    });
  };

  // Handle para enviar feedback
  const handleSendFeedback = async () => {
    console.log("Botao de enviar feedback pressionado");

    setIsLoading(true);

    let feedbackData = {};

    if (state_data_collection) {
      feedbackData = {
        source_app: 0, // Aplicativo mobile
        timestamp: timestamp,
        language: topLanguage,
        ortography: topOrthography || "undefined",
        language_model: null,
        overall_feedback: generalFeedback,
        translation_feedback: translationFeedback,
        spelling_feedback: correctionFeedback,
        prediction_feedback: suggestionFeedback,
        user: username,
        translation_log_entries: translationLogs,
        spelling_log_entries: correctionLogs,
        prediction_log_entries: suggestionLogs,
        config_flags: {
          disable_dic: disableDic,
          disable_next: disableNext,
          disable_word_meaning: disableWordMeaning,
        },
      };
    } else {
      feedbackData = {
        source_app: 0, // Aplicativo mobile
        timestamp: timestamp,
        language: topLanguage,
        ortography: topOrthography || "undefined",
        language_model: null,
        overall_feedback: generalFeedback,
        translation_feedback: translationFeedback,
        spelling_feedback: correctionFeedback,
        prediction_feedback: suggestionFeedback,
        user: username,
        translation_log_entries: [],
        spelling_log_entries: [],
        prediction_log_entries: [],
        config_flags: {
          disable_dic: disableDic,
          disable_next: disableNext,
          disable_word_meaning: disableWordMeaning,
        },
      };
    }

    console.log(JSON.stringify(feedbackData));
    try {
      await axios.post(
        apiUrl + "/add_feedback",
        feedbackData,
        {
          headers: headerS,
        }
      );

      // limpa context e global
      clearFeedbackData();
      suggestionData.length = 0;

      // ao navegar, garanta que o username original seja repassado
      navigation.navigate("MainPage", {
        resetFeedback: true,
        username: username,
      });
    } catch (error) {
      console.error("Erro ao enviar feedback:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Estilo do título dos blocos
  const titleStyle = {
    fontSize: 20,
    textAlign: "center" as const,
    marginBottom: 16,
    color: "#666666",
    fontWeight: "500" as const,
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#FFFFFF" }}>
      <StatusBar barStyle="light-content" backgroundColor="#0F62FE" />
      <View
        style={{
          backgroundColor: "#0F62FE",
          paddingVertical: 8,
          paddingHorizontal: 12,
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Text style={{ color: "#FFF", fontSize: 18, fontWeight: "600" }}>
          Pinimasa yẽgatu rupi
        </Text>
        <TouchableOpacity onPress={handleNavigateToMainPage}>
          <Text style={{ color: "#FFF", fontSize: 18, fontWeight: "600" }}>
            Voltar
          </Text>
        </TouchableOpacity>
      </View>
      <View
        style={{
          flex: 1,
          margin: 8,
          padding: 8,
          borderWidth: 2,
          borderColor: "#0F62FE",
          borderRadius: 8,
        }}
      >
        {/* Geral */}
        <View style={{ alignItems: "center", padding: 20 }}>
          <Text style={titleStyle}>Avalie o aplicativo</Text>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-around",
              width: "70%",
            }}
          >
            <TouchableOpacity onPress={() => setGeneralFeedback("Positive")}>
              <Image
                source={
                  generalFeedback === "Positive"
                    ? thumbsUpFilled
                    : thumbsUpOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setGeneralFeedback("Negative")}>
              <Image
                source={
                  generalFeedback === "Negative"
                    ? thumbsDownFilled
                    : thumbsDownOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
          </View>
        </View>
        {/* Tradução */}
        <View style={{ alignItems: "center", padding: 20 }}>
          <Text style={titleStyle}>Avalie a tradução</Text>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-around",
              width: "70%",
            }}
          >
            <TouchableOpacity
              onPress={() => setTranslationFeedback("Positive")}
            >
              <Image
                source={
                  translationFeedback === "Positive"
                    ? thumbsUpFilled
                    : thumbsUpOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => setTranslationFeedback("Negative")}
            >
              <Image
                source={
                  translationFeedback === "Negative"
                    ? thumbsDownFilled
                    : thumbsDownOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
          </View>
        </View>
        {/* Correção */}
        <View style={{ alignItems: "center", padding: 20 }}>
          <Text style={titleStyle}>Avalie a correção ortográfica</Text>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-around",
              width: "70%",
            }}
          >
            <TouchableOpacity onPress={() => setCorrectionFeedback("Positive")}>
              <Image
                source={
                  correctionFeedback === "Positive"
                    ? thumbsUpFilled
                    : thumbsUpOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setCorrectionFeedback("Negative")}>
              <Image
                source={
                  correctionFeedback === "Negative"
                    ? thumbsDownFilled
                    : thumbsDownOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
          </View>
        </View>
        {/* Sugestão */}
        <View style={{ alignItems: "center", padding: 20 }}>
          <Text style={titleStyle}>Avalie a sugestão de palavras</Text>
          <View
            style={{
              flexDirection: "row",
              justifyContent: "space-around",
              width: "70%",
            }}
          >
            <TouchableOpacity onPress={() => setSuggestionFeedback("Positive")}>
              <Image
                source={
                  suggestionFeedback === "Positive"
                    ? thumbsUpFilled
                    : thumbsUpOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setSuggestionFeedback("Negative")}>
              <Image
                source={
                  suggestionFeedback === "Negative"
                    ? thumbsDownFilled
                    : thumbsDownOutline
                }
                style={{ width: 50, height: 50 }}
              />
            </TouchableOpacity>
          </View>
        </View>
        {/* Botão enviar */}
        <EnviandoFeedbackButton
          title={isLoading ? "Enviando..." : "Enviar o feedback"}
          onPress={handleSendFeedback}
          disabled={isLoading}
          accessibilityLabel="Botão para enviar feedback"
          style={{
            position: "absolute",
            bottom: 8,
            left: 20,
            right: 20,
          }}
        />
      </View>
    </SafeAreaView>
  );
};

export default FeedbackPage;
