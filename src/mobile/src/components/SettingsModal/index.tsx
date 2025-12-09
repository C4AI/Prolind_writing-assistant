import AsyncStorage from "@react-native-async-storage/async-storage";
import { useNavigation } from "@react-navigation/native";
import React, { SetStateAction, useEffect, useState } from "react";
import {
  View,
  Text,
  Modal,
  Switch,
  ScrollView,
  Pressable,
  Alert,
} from "react-native";
import RNFS from "react-native-fs";
import SVGIcon from "../SVGIcon";
import infoIcon from "../../assets/info-icon";
import { responsiveFontSize } from "../../utils/FontContext";
import { useFeedback } from "../../contexts/FeedbackContext";

interface SettingsModalProps {
  isVisible: boolean;
  onClose: () => void;
  settings: {
    enableDictionary: boolean;
    enableNextWordSuggestion: boolean;
    enableWordMeaning: boolean;
    enableDataCollection: boolean;
  };
  onSettingChange: (setting: string, value: boolean) => void;
  language: string;
}

const uiText = {
  pt: {
    title: "Configurações",
    dictionary: "Sugestão de palavra",
    nextWord: "Sugestão de próxima palavra",
    wordMeaning: "Tradução da palavra",
    dataCollection: "Coleta de dados",
    version: "Versão",
    exit: "Sair",
    info: {
      dictionary:
        "Sugere maneiras de completar a palavra que está sendo digitada.",
      nextWord: "Sugere a próxima palavra da sentença sendo digitada.",
      wordMeaning:
        "Exibe o significado traduzido da palavra na sugestão de palavras e na sugestão de próxima palavra.",
      dataCollection: "Permite a coleta de dados para análise",
    },
  },
  en: {
    title: "Settings",
    dictionary: "Suggestion of word",
    nextWord: "Next word suggestion",
    wordMeaning: "Word translation",
    dataCollection: "Data collection",
    version: "Version",
    exit: "Exit",
    info: {
      dictionary: "Suggests ways to complete the word being typed.",
      nextWord: "Suggests the next word in the sentence being typed.",
      wordMeaning:
        "Displays the translated meaning of words in word suggestion.",
      dataCollection: "Allows data collection for analysis",
    },
  },
};

const SettingsModal: React.FC<SettingsModalProps> = ({
  isVisible,
  onClose,
  settings,
  onSettingChange,
  language,
}) => {
  const [version, setVersion] = useState<string>("");
  const [localSettings, setLocalSettings] = useState(settings);
  const navigation = useNavigation();
  const { isFeedbackEnabled } = useFeedback();

  const lang = language === "Inglês" ? "en" : "pt";

  const getText = (key: string): string => uiText[lang][key];
  const getInfoText = (key: string): string => uiText[lang].info[key];

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
    const loadSettings = async () => {
      try {
        const savedSettings = await AsyncStorage.getItem("settings");
        if (savedSettings) {
          setLocalSettings(JSON.parse(savedSettings));
        }
      } catch (err) {
        console.log("Erro ao carregar configurações:", err);
      }
    };

    loadSettings();
  }, []);

  const handleLogout = async () => {
    try {
      await AsyncStorage.removeItem("username");
      await AsyncStorage.removeItem("password");
      navigation.navigate("Login");
    } catch (error) {
      Alert.alert("Erro", "Ocorreu um erro ao realizar o logout.");
    }
  };

  const handleSettingChange = async (settingKey: string, value: boolean) => {
    const updatedSettings = { ...localSettings, [settingKey]: value };
    setLocalSettings(updatedSettings);

    try {
      await AsyncStorage.setItem("settings", JSON.stringify(updatedSettings));
    } catch (err) {
      console.log("Erro ao salvar configurações:", err);
    }

    onSettingChange(settingKey, value);
  };

  const renderSetting = (
    title: string,
    value: boolean,
    settingKey: string,
    getTextString: string
  ) => (
    <View className="flex-row justify-between items-center py-3 border-b border-[#C1C7CD]">
      <View className="flex-row items-center w-[70%]">
        <Pressable
          className="mr-1"
          onPress={() => Alert.alert(title, getInfoText(getTextString))}
        >
          <SVGIcon xml={infoIcon} width={20} height={20} color="#0F62FE" />
        </Pressable>
        <Text
          className={` text-[#000000] mr-2`}
          style={{ fontSize: responsiveFontSize(16) }}
        >
          {title}
        </Text>
      </View>
      <Switch
        value={value}
        onValueChange={(newValue) => handleSettingChange(settingKey, newValue)}
        trackColor={{ false: "#767577", true: "#0F62FE" }}
        thumbColor={value ? "#ffffff" : "#f4f3f4"}
        ios_backgroundColor="#3e3e3e"
      />
    </View>
  );

  return (
    <Modal
      className="h-full"
      visible={isVisible}
      transparent
      animationType="fade"
    >
      <Pressable
        className="h-full justify-center items-center"
        style={{ backgroundColor: "rgba(0,0,0,0.8)" }}
        onPress={onClose}
      >
        <View className="bg-white rounded-t-lg w-4/5 pb-2">
          <View className="bg-[#0F62FE] py-2 px-3">
            <Text
              className={`text-white font-semibold`}
              style={{ fontSize: responsiveFontSize(16) }}
            >
              {getText("title")}
            </Text>
          </View>

          <ScrollView className="px-3" persistentScrollbar={true}>
            {renderSetting(
              getText("dictionary"),
              localSettings.enableDictionary,
              "enableDictionary",
              "dictionary"
            )}
            {renderSetting(
              getText("nextWord"),
              localSettings.enableNextWordSuggestion,
              "enableNextWordSuggestion",
              "nextWord"
            )}
            {renderSetting(
              getText("wordMeaning"),
              localSettings.enableWordMeaning,
              "enableWordMeaning",
              "wordMeaning"
            )}
            {isFeedbackEnabled &&
              renderSetting(
                getText("dataCollection"),
                settings.enableDataCollection,
                "enableDataCollection",
                "dataCollection"
              )}

            <View className="justify-center items-center flex flex-row mt-2">
              <Pressable
                className={`px-4 py-1 rounded-md border-2 border-[#0F62FE] bg-[#0F62FE]`}
                onPress={handleLogout}
              >
                <Text
                  style={{ fontSize: responsiveFontSize(14) }}
                  className="text-white"
                >
                  {getText("exit")}
                </Text>
              </Pressable>
              <Text
                className={`font-semibold my-2 ml-auto`}
                style={{ fontSize: responsiveFontSize(16) }}
              >
                {getText("version")} {version}
              </Text>
            </View>
          </ScrollView>
        </View>
      </Pressable>
    </Modal>
  );
};

export default SettingsModal;
