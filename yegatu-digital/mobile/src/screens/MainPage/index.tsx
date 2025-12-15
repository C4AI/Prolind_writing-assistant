import React, { useState, useRef, useEffect } from "react";
import {
  Image,
  Share,
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TextInput,
  Keyboard,
  TouchableWithoutFeedback,
  Button,
  NativeSyntheticEvent,
  TextInputSelectionChangeEventData,
  Dimensions,
} from "react-native";
import Clipboard from "@react-native-clipboard/clipboard";
import NetInfo from "@react-native-community/netinfo";
import SVGIcon from "../../components/SVGIcon";
import settingsIconSvg from "../../assets/settings";
import refreshCwIconSvg from "../../assets/refresh-cw";
import arrowRightIcon from "../../assets/arrow-right";
import feedbackIcon from "../../assets/feedback.png";
import LanguageSelector from "../../components/LanguageSelector";
import TranslationBox from "../../components/TranslationBox";
import ActionButtons from "../../components/ActionButtons";
import SpecialCharacters from "../../components/SpecialCharacters";
import SuggestionBox from "../../components/SuggestionBox";
import SettingsModal from "../../components/SettingsModal";
import WordMeaningBox from "../../components/WordMeaningBox";
import CorrectionSuggestionBox from "../../components/CorrectionSuggestionBox";
import NoInternetPopup from "../../components/NoInternetPopup";
import { targetLanguages } from "../../config";
import { useAuth } from "../../utils/AuthContext";
import {
  BodyPt,
  BodyYrl,
  BodyEn,
  BodyYrlEn,
  LanguageHeaders,
  BodyYrlConvert,
  LanguageTopology,
} from "../../types";
import axios, { AxiosError, isAxiosError } from "axios";
import { Language, LanguagesData } from "../../interfaces";
import LoadingOverlay from "../../components/LoadingOverlay";
import FeedbackModal from "../../components/FeedbackModal";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { responsiveFontSize } from "../../utils/FontContext";
// GI: Import the suggestionLog from the SuggestionBox component
import { suggestionData } from "../../components/SuggestionBox";
import { useNavigation, useRoute } from "@react-navigation/native";
import FeedbackButton from "../../components/FeedbackButton";
import LogLimitPopup from "../../components/LogLimitPopup";
import { useFeedback } from "../../contexts/FeedbackContext";
import DataCapturePopup from "../../components/DataCapturePopup";
import { useLogPopupHandler } from "./handlers/useLogPopupHandler";

// Main page component for the writing assistant app
const MainPage: React.FC = () => {
  const route = useRoute();

  const {
    setFeedbackData,
    isFeedbackEnabled,
    feedbackCountThreshold,
    feedbackTimeThreshold,
    setFeedbackCountThreshold,
    setFeedbackTimeThreshold,
    setIsFeedbackEnabled,
  } = useFeedback();
  const [dataCaptureVisible, setDataCaptureVisible] = useState(true);

  const handleCloseDataCapture = () => {
    setDataCaptureVisible(false);
  };

  const username = (route.params as { username?: string })?.username || "";

  //LAIZ: Log de correction word.

  const date = new Date();
  const timestamp = date.getTime();

  //LAIZ: The navigation variable retrieves the FeedbackPage component and manages the loading state.
  const navigation = useNavigation();
  const [isLoading, setIsLoading] = useState(false);

  const correctionLogs = useRef<any[]>([]);

  const addCorrectionLog = (log: any) => {
    correctionLogs.current.push(log);
  };

  //ISA: Set the list of translation logs, now using useRef
  const translationLogs = useRef<any[]>([]);
  //ISA: Add a translation log to the list
  const addTranslationLog = (log: any) => {
    translationLogs.current.push(log);
  };

  // GI: Set the list of suggestion logs, now using useRef
  const suggestionLogs = useRef<any[]>([]);
  // GI: Add a suggestion log to the list
  const addSuggestionLog = (log: any) => {
    suggestionLogs.current.push(log);
  };

  // GI: Clean the suggestion logs
  const cleanSuggestionLogs = () => {
    suggestionLogs.current = [];
  };

  // First, we define the getters and setters for the state variables we will use in the component

  const [isIndigenousOnTop, setIsIndigenousOnTop] = useState(true);
  const [topLanguage, setTopLanguage] = useState({
    name: "Nheengatu",
    orthography: "arn1",
  });
  const [bottomLanguage, setBottomLanguage] = useState({
    name: "Português",
    orthography: "",
  });
  const isBothLanguagesIndigenous =
    topLanguage.name === "Nheengatu" && bottomLanguage.name === "Nheengatu";
  const [languagesTopology, setLanguagesTopology] = useState<
    LanguageTopology[]
  >([]);

  const [chars, setChars] = useState([]);

  useEffect(() => {
    const loadLanguageSettings = async () => {
      try {
        const savedTopLanguage = await AsyncStorage.getItem("topLanguage");
        const savedTopOrthography =
          await AsyncStorage.getItem("topOrthography");

        if (savedTopLanguage && savedTopOrthography) {
          console.log(savedTopLanguage);
          console.log(savedTopOrthography);
          setTopLanguage({
            name: savedTopLanguage,
            orthography: savedTopOrthography,
          });
        } else {
          try {
            const response = await axios.get(apiUrl + "/languages", {
              headers: headerS,
            });
            setTopLanguage({
              name: response.data["language"],
              orthography: response.data["orthography"],
            });
          } catch (error) {
            console.error("Erro ao buscar os dados:", error);
          }
        }
      } catch (error) {
        console.error("Erro ao carregar as configurações de língua:", error);
      }
    };

    loadLanguageSettings();
  }, []);

  const canTranslate = () => {
    const topOrthography = topLanguage.orthography;
    const topLanguageName = topLanguage.name;
    const bottomOrthography = bottomLanguage.orthography;
    const bottomLanguageName = bottomLanguage.name;
    let canTranslate = false;
    languagesTopology.map((languageTopology) => {
      if (languageTopology.name === topLanguageName) {
        languageTopology.ortographiesTopology?.map((orthographyTopology) => {
          if (orthographyTopology.name === topOrthography) {
            canTranslate = orthographyTopology.translatorEnabled;
          }
        });
      }
    });
    languagesTopology.map((languageTopology) => {
      if (languageTopology.name === bottomLanguageName) {
        languageTopology.ortographiesTopology?.map((orthographyTopology) => {
          if (orthographyTopology.name === bottomOrthography) {
            canTranslate = orthographyTopology.translatorEnabled;
          }
        });
      }
    });
    return canTranslate;
  };

  const canCorrrect = () => {
    const topOrthography = topLanguage.orthography;
    const topLanguageName = topLanguage.name;
    let canCorrect = false;
    languagesTopology.map((languageTopology) => {
      if (languageTopology.name === topLanguageName) {
        languageTopology.ortographiesTopology?.map((orthographyTopology) => {
          if (orthographyTopology.name === topOrthography) {
            canCorrect = orthographyTopology.spellCheckerEnabled;
          }
        });
      }
    });
    return canCorrect;
  };

  const canSuggestNextWord = () => {
    const topOrthography = topLanguage.orthography;
    const topLanguageName = topLanguage.name;
    let canSuggestNextWord = false;
    languagesTopology.map((languageTopology) => {
      if (languageTopology.name === topLanguageName) {
        languageTopology.ortographiesTopology?.map((orthographyTopology) => {
          if (orthographyTopology.name === topOrthography) {
            canSuggestNextWord = orthographyTopology.nextWordEnabled;
          }
        });
      }
    });
    return canSuggestNextWord;
  };

  const haveDictionary = () => {
    const topOrthography = topLanguage.orthography;
    const topLanguageName = topLanguage.name;
    let haveDictionary = false;
    languagesTopology.map((languageTopology) => {
      if (languageTopology.name === topLanguageName) {
        languageTopology.ortographiesTopology?.map((orthographyTopology) => {
          if (orthographyTopology.name === topOrthography) {
            haveDictionary = orthographyTopology.dictionaryEnabled;
          }
        });
      }
    });
    return haveDictionary;
  };

  // This object contains the text for the UI in both Portuguese and English, which will be used to display messages to the user
  const uiText = {
    pt: {
      appTitle: "Pinimasa yẽgatu rupi",
      inputPlaceholder: "Digite o texto",
      translatedText: "Texto traduzido:",
      translationPlaceholder: "Tradução",
      checkButton: "Checar ortografia",
      checkingButton: "Checando...",
      examineButton: "Ver significados",
      translateButton: "Traduzir",
      convertButton: "Converter",
      translatingButton: "Traduzindo...",
      convertingButton: "Convertendo...",
      copySuccess: {
        title: "Texto copiado",
        message: "O texto foi copiado para a área de transferência.",
      },
      errors: {
        translation: {
          timeout:
            "A tradução excedeu o tempo limite. Por favor, tente novamente.",
          generic: "Erro ao traduzir o texto. Por favor, tente novamente.",
        },
        correction: {
          timeout:
            "A correção excedeu o tempo limite. Por favor, tente novamente.",
          generic: "Erro ao corrigir o texto. Por favor, tente novamente.",
        },
      },
      correction: {
        success: {
          title: "Texto Correto",
          message: "Não foram encontradas sugestões de correção no texto!",
        },
      },
      error: {
        title: "Erro",
      },
    },
    en: {
      appTitle: "Writing Assistant",
      inputPlaceholder: "Type your text",
      translatedText: "Translated text:",
      translationPlaceholder: "Translation",
      checkButton: "Check spelling",
      checkingButton: "Checking...",
      examineButton: "See meanings",
      translateButton: "Translate",
      convertButton: "Convert",
      translatingButton: "Translating...",
      convertingButton: "Converting...",
      copySuccess: {
        title: "Text copied",
        message: "The text has been copied to the clipboard.",
      },
      errors: {
        translation: {
          timeout: "Translation timed out. Please try again.",
          generic: "Error translating text. Please try again.",
        },
        correction: {
          timeout: "Correction timed out. Please try again.",
          generic: "Error correcting text. Please try again.",
        },
      },
      correction: {
        success: {
          title: "Text is Correct",
          message: "No correction suggestions were found in the text!",
        },
      },
      error: {
        title: "Error",
      },
    },
  };

  // This function is used to get the text for the UI based on the current language and the path to the text in the object
  const getText = (path: string): string => {
    /*
    This function receives a path to the text in the object and returns the text in the current language.
    The function first checks if the indigenous language is on top, and then gets the target language based on that.
    The function then gets the text from the object using the target language and the path.
    */
    const targetLanguage = isIndigenousOnTop
      ? bottomLanguage.name
      : topLanguage.name;
    const language = targetLanguage === "Inglês" ? "en" : "pt";

    const pathArray = path.split(".");
    let current: any = uiText[language];

    for (const key of pathArray) {
      current = current[key];
      if (current === undefined) return path;
    }

    return current;
  };

  const targetLanguage = isIndigenousOnTop
    ? bottomLanguage.name
    : topLanguage.name;

  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [currentWord, setCurrentWord] = useState("");
  const [lastChar, setLastChar] = useState("");
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const inputRef = useRef<TextInput>(null);
  const [isSettingsModalVisible, setIsSettingsModalVisible] = useState(false);
  const { token, setToken } = useAuth();
  const [wordMeanings, setWordMeanings] = useState<string[]>([]);
  const { tokenEn, setTokenEn } = useAuth();
  const [isInputFocused, setIsInputFocused] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingCorrection, setLoadingCorrection] = useState(false);
  const [settings, setSettings] = useState({
    enableDictionary: true,
    enableNextWordSuggestion: true,
    enableWordMeaning: true,
    enableDataCollection: true,
  });
  const [spell_checker_timeout, setSpell_checker_timeout] = useState(60000);
  const [translator_timeout, setTranslator_timeout] = useState(60000);
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(true);
  const apiUrl =
  //  "https://assistente-escrita-linguas-indigenas-backend.y6dbcklf96p.us-south.codeengine.appdomain.cloud";
  //  "http://10.0.2.2:3000";
    "http://104.154.155.83";

  const [languages, setLanguages] = useState();
  const [indigenousLanguages, setIndigenousLanguages] = useState(["Nheengatu"]);
  type LanguageHeaders = {
    [key: string]: {
      [key: string]: {
        Authorization: string | null;
        "Content-Type": string;
        accept: string;
      };
    };
  };
  const [selectedCorrectionWord, setSelectedCorrectionWord] = useState<
    string | null
  >(null);
  const [isCorrectionActive, setIsCorrectionActive] = useState(false);
  const [isWordMeaningSearchActive, setIsWordMeaningSearchActive] =
    useState(false);
  const headerS = {
    Authorization: token,
    "Content-Type": "application/json",
    accept: "application/json",
  };

  const [dictionaryEntries, setDictionaryEntries] = useState<{
    dic_words: { [key: string]: string[] };
    char_conv_map: string[][];
  }>({
    dic_words: {},
    char_conv_map: [],
  });

  const [dictionaryEntriesEn, setDictionaryEntriesEn] = useState<{
    dic_words: { [key: string]: string[] };
    char_conv_map: string[][];
  }>({
    dic_words: {},
    char_conv_map: [],
  });

  const [feedbackModal, setFeedbackModal] = useState({
    visible: false,
    title: "",
    message: "",
    type: "info" as "error" | "success" | "info" | "warning",
    isTempPopUp: false,
  });

  const [lastSavedLanguage, setLastSavedLanguage] = useState("");
  const [lastSavedOrthography, setLastSavedOrthography] = useState("");

  const showFeedback = (
    title: string,
    message: string,
    type: "error" | "success" | "info" | "warning" = "info",
    isTempPopUp?: boolean
  ) => {
    /*
    This function receives the title, message, type, and isTempPopUp as parameters and sets the feedbackModal state variable with the received values.
    */
    setFeedbackModal({
      visible: true,
      title,
      message,
      type,
      isTempPopUp: isTempPopUp || false,
    });
  };

  const showTempPopUp = (title: string, message: string) => {
    /*
    This function receives the title and message as parameters and calls the showFeedback function with the received values and the isTempPopUp set to true.
    It is used to show temporary pop-ups that disappear after a few seconds.
    */
    showFeedback(title, message, "info", true);
  };

  const showError = (message: string) => {
    /*
    This function receives the message as a parameter and calls the showFeedback function with the title set to "Error" and the message and type set to the received values.
    */
    showFeedback(getText("error.title"), message, "error");
  };

  // These state variables are used to store the configuration for the axios requests to the spell checker and translator APIs
  const [axiosSpellConfig, setAxiosSpellConfig] = useState({
    timeout: 30000,
    headers: headerS,
  });

  const [axiosTranslateConfig, setAxiosTranslateConfig] = useState({
    timeout: 30000,
    headers: headerS,
  });

  // This is a type definition for the CorrectionInfo object, which contains information about a correction suggestion
  // It has the word, suggestion, position, and allSuggestions properties
  // It was created to make sure that the accepted correction suggestions go into the right position in the text
  type CorrectionInfo = {
    word: string;
    suggestion: string;
    position: number;
    allSuggestions: string[];
  };

  const [correctionData, setCorrectionData] = useState<
    Record<string, CorrectionInfo>
  >({});
  const [selectedCorrectionPosition, setSelectedCorrectionPosition] = useState<
    number | null
  >(null);

  // These Mapper objects are used to map the language names to the URLs and output fields in the API responses
  // When new languages are added, they should be added to these objects as well
  const LanguageToUrlMapper: {
    [key: string]: { [key: string]: string };
  } = {
    Nheengatu: {
      Português: "translate_yrl",
      Inglês: "translate_yrl_en",
      Nheengatu: "convert_yrl",
    },
    Português: { Nheengatu: "translate_pt" },
    Inglês: { Nheengatu: "translate_en" },
  };

  const LanguageToOutputFieldMapper: {
    [key: string]: { [key: string]: string };
  } = {
    Nheengatu: {
      Português: "sentence_pt",
      Inglês: "sentence_en",
      Nheengatu: "converted_sentence",
    },
    Português: { Nheengatu: "sentence_yrl" },
    Inglês: { Nheengatu: "sentence_yrl" },
  };

  const bodyMapper = {
    Português: (data: BodyPt) => ({
      sentence_pt: data.sentence_pt,
      ortography: data.ortography,
      disable_dic: data.disable_dic,
      disable_next: data.disable_next,
      disable_word_meaning: data.disable_word_meaning,
    }),
    Nheengatu: (data: BodyYrl) => ({
      sentence_yrl: data.sentence_yrl,
      ortography: data.ortography,
      selected_next_words: data.selected_next_words,
      selected_dict_words: data.selected_dict_words,
      disable_dic: data.disable_dic,
      disable_next: data.disable_next,
      disable_word_meaning: data.disable_word_meaning,
    }),
    Inglês: (data: BodyEn) => ({
      sentence_en: data.sentence_en,
      ortography: data.ortography,
      disable_dic: data.disable_dic,
      disable_next: data.disable_next,
      disable_word_meaning: data.disable_word_meaning,
    }),
    NheengatuEn: (data: BodyYrlEn) => ({
      sentence_yrl: data.sentence_yrl,
      ortography: data.ortography,
      selected_next_words: data.selected_next_words,
      selected_dict_words: data.selected_dict_words,
      disable_dic: data.disable_dic,
      disable_next: data.disable_next,
      disable_word_meaning: data.disable_word_meaning,
    }),
    NheengatuConvert: (data: BodyYrlConvert) => ({
      sentence: data.sentence,
    }),
  };

  const saveLanguagePreference = async (
    language: string,
    orthography: string | undefined
  ) => {
    /*
    This function receives the language and orthography as parameters and sends a POST request to the API to save the language preference.
    It first checks if the language or orthography have changed since the last time they were saved.
    If they have changed, it sends the request to the API to save the new language preference.
    */
    if (
      language !== lastSavedLanguage ||
      orthography !== lastSavedOrthography
    ) {
      try {
        await axios.post(
          `${apiUrl}/change_language`,
          {
            language: language,
            orthography: orthography || "",
          },
          { headers: headerS }
        );

        setLastSavedLanguage(language);
        setLastSavedOrthography(orthography || "");
      } catch (error) {
        console.error("Erro ao atualizar preferência de língua:", error);
      }
    } else {
    }
  };

  // Variavel para controlar a exibição do popup de log

  const getEnabledLanguages = (languagesData: LanguagesData): string[] => {
    /*
    This function receives the languagesData as a parameter and returns an array with the names of the enabled languages.
    It filters the languagesData object to get the languages that are enabled and returns an array with their names.
    */
    return Object.keys(languagesData).filter(
      (language) => languagesData[language].enabled
    );
  };

  const handleFetchLanguageUrls = async (language: string) => {
    console.log("Língua" + language);
    try {
      await axios.get(`${apiUrl}/language_urls?language=${language}`, {
        headers: headerS,
      });
    } catch (error) {
      console.error("Erro ao buscar os dados 3:", error);
      if (isAxiosError(error)) {
      }
    }
  };

  const fetchDictionary = async (language: string, orthography: string) => {
    /*
    This function sends a GET request to the API to fetch the dictionary entries for the indigenous language in Portuguese.
    It sets the dictionary entries in the state variable when the data is received.
    */
    try {
      const response = await axios.post(apiUrl + "/dic_words", {
        headers: headerS,
        language: language,
        orthography: orthography.toUpperCase(),
      });
      if (response.data.dic_words && response.data.char_conv_map) {
        setDictionaryEntries(response.data);
      }
    } catch (error) {
      console.error("Erro ao buscar os dados 1:", error);
    }
  };

  const fetchDictionaryEn = async (language: string, orthography: string) => {
    /*
    This function sends a GET request to the API to fetch the dictionary entries for the indigenous language in English.
    It sets the dictionary entries in the state variable when the data is received.
    */
    try {
      console.log("ORTOGRAFIA: " + orthography);
      const response = await axios.post(apiUrl + "/dic_words_en", {
        headers: headerS,
        language: language,
        orthography: orthography.toUpperCase(),
      });
      if (response.data.dic_words && response.data.char_conv_map) {
        setDictionaryEntriesEn(response.data);
      }
    } catch (error) {
      console.error("Erro ao buscar os dados 2:", error);
    }
  };

  // This useEffect hook is used to fetch the languages, dictionary entries, and timeout settings from the API when the component mounts
  // It also sets up event listeners for the keyboard show and hide events
  // It only runs once when the component mounts, so it is used to fetch the initial data from the API
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      /*
      This function sets the isConnected state variable when the connection status changes.
      It is used to show a pop-up when the user is not connected to the internet.
      */
      setIsConnected(state.isConnected);
    });

    const fetchLanguages = async () => {
      /*
      This function sends a GET request to the API to fetch the languages and orthographies available for translation.
      It sets the languages and orthographies in the state variables when the data is received.
      */
      try {
        const response = await axios.get(apiUrl + "/languages", {
          headers: headerS,
        });
        setBottomLanguage({ name: "Português", orthography: "" });
        setLanguages(response.data["languages"]);
        setIndigenousLanguages(getEnabledLanguages(response.data.languages));

        setLastSavedLanguage(response.data["language"]);
        setLastSavedOrthography(response.data["orthography"]);
      } catch (error) {
        console.error("Erro ao buscar os dados:", error);
      }
    };

    const fetchTimeout = async () => {
      /*
      This function sends a GET request to the API to fetch the timeout settings for the spell checker and translator.
      It sets the timeout settings in the state variables when the data is received
      */
      try {
        const response = await axios.get(apiUrl + "/timeout", {
          headers: headerS,
        });
        const { spell_checker_timeout, translator_timeout } = response.data;

        /* setAxiosSpellConfig((prevConfig) => ({
          ...prevConfig,
          timeout: spell_checker_timeout * 10000,
        })); */

        setAxiosTranslateConfig((prevConfig) => ({
          ...prevConfig,
          timeout: 30000,
        }));
      } catch (error) {
        console.error("Erro ao buscar os dados:", error);
      }
    };

    fetchTimeout();
    fetchLanguages();

    return () => {
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    handleFetchLanguageUrls(topLanguage.name);
    if (topLanguage.orthography === "arn1") {
      return;
    }
    fetchDictionary(topLanguage.name, topLanguage.orthography);
    fetchDictionaryEn(topLanguage.name, topLanguage.orthography);
  }, [topLanguage]);

  useEffect(() => {
    const handleLanguagesTopology = async () => {
      try {
        const response = await axios.get(`${apiUrl}/languages_topology`, {
          headers: headerS,
        });
        setLanguagesTopology(response.data);
      } catch (error) {
        console.error("Erro ao buscar os dados 3:", error);
        if (isAxiosError(error)) {
        }
      }
    };
    handleLanguagesTopology();
  }, []);

  const saveLanguageSettings = async (
    language: string,
    orthography: string
  ) => {
    try {
      await AsyncStorage.setItem("topLanguage", language);
      await AsyncStorage.setItem("topOrthography", orthography);
    } catch (error) {
      console.error("Erro ao salvar configurações de língua", error);
    }
  };

  const handleLanguageChange = async (
    selectedLanguage: string,
    selectedOrthography: string
  ) => {
    setTopLanguage({
      name: selectedLanguage,
      orthography: selectedOrthography,
    });

    await saveLanguageSettings(selectedLanguage, selectedOrthography);
  };

  const toggleLanguages = () => {
    /*
    This function is used to toggle the languages between the top and bottom positions.
    It swaps the top and bottom languages, sets the input text to the output text, and vice versa.
    It also resets the selected word, disables the word meaning search, and disables the correction mode.
    */
    let bottomLanguageChangeble = bottomLanguage;
    if (isIndigenousOnTop && bottomLanguage.name === "Nheengatu") {
      bottomLanguageChangeble = {
        name: "Português",
        orthography: "",
      };
    }
    setTopLanguage(bottomLanguageChangeble);
    setBottomLanguage(topLanguage);
    setIsIndigenousOnTop(!isIndigenousOnTop);
    setInputText(outputText);
    setOutputText(inputText);

    setSelectedWord(null);
    disableWordMeaningSearch();
    disableCorrection();
  };

  const getSpecialCharacters = () => {
    /*
    This function returns the special characters for the current language.  
    It checks if the indigenous language is on top and if the top language is Nheengatu with an orthography selected.
    If both conditions are met, it returns the special characters for the selected orthography.
    Otherwise, it returns an empty array
    */
    if (
      isIndigenousOnTop &&
      topLanguage.orthography &&
      topLanguage.name &&
      languages &&
      languages[topLanguage.name] &&
      languages[topLanguage.name].ortographies &&
      languages[topLanguage.name].ortographies[topLanguage.orthography] &&
      languages[topLanguage.name].ortographies[topLanguage.orthography][
        "special_chars"
      ]
    ) {
      setChars(
        languages[topLanguage.name].ortographies[topLanguage.orthography][
          "special_chars"
        ]
      );
      return;
    }
    setChars([]);
    return;
  };

  const [selection, setSelection] = useState({ start: 0, end: 0 });

  useEffect(() => {
    const getFeedbackConfig = async () => {
      try {
        console.log("TESTE");
        const response = await axios.get(apiUrl + "/get_feedback_config", {
          headers: headerS,
        });

        console.log("Get feedback config");
        console.log("TESTE:  " + response.data["enable_feedback"]);
        setFeedbackTimeThreshold(response.data["feedback_time_threshold"]);
        setIsFeedbackEnabled(response.data["enable_feedback"]);
        setFeedbackCountThreshold(response.data["feedback_count_threshold"]);
      } catch (e) {
        console.log(e);
      }
    };

    getFeedbackConfig();
  }, []);

  const handleSelectionChange = (
    event: NativeSyntheticEvent<TextInputSelectionChangeEventData>
  ) => {
    setSelection(event.nativeEvent.selection);
  };

  const insertSpecialCharacter = (char: string) => {
    // Get cursor position from current selection state
    const cursorPosition = selection.start;

    // Create new text by inserting character at cursor position
    const newText =
      inputText.slice(0, cursorPosition) +
      char +
      inputText.slice(cursorPosition);

    // Update input text
    setInputText(newText);
    handleTextChange(newText);

    // Update selection state to move cursor after inserted character
    const newPosition = cursorPosition + 1;
    setSelection({ start: newPosition, end: newPosition });

    // Update the input component
    if (inputRef.current) {
      inputRef.current.setNativeProps({
        text: newText,
        selection: { start: newPosition, end: newPosition },
      });
    }
  };

  const handleTextChange = (text: string) => {
    console.log(text);

    /*
    This function receives the text as a parameter and sets the input text in the state variable.
    It also sets the current word and last character in the state variables.
    If the word meaning search is active and the text is not empty, it disables the word meaning search
    */
    setInputText(text);
    const words = text.split(/\s+/);
    setCurrentWord(words[words.length - 1]);
    setLastChar(text[text.length - 1] || "");

    if (isWordMeaningSearchActive && text.trim() !== "") {
      disableWordMeaningSearch();
    }
  };

  const handleSuggestionPress = (suggestion: string) => {
    /*
    This function receives the suggestion as a parameter and inserts the suggestion in the input text.
    It first removes any numbers at the end of the suggestion and the last character of the input text.
    It then checks if the last character is a space or punctuation and adds the suggestion to the input text accordingly.
    If the last character is not a space or punctuation, it replaces the last word in the input text with the suggestion.
    */

    const cleanSuggestion = suggestion.replace(/\(\d+\)$/, "");
    // Get cursor position from current selection state
    const cursorPosition = selection.start;

    let newText;
    if (lastChar === " " || /[.,!?]/.test(lastChar)) {
      newText = inputText + cleanSuggestion;
    } else {
      const words = inputText.split(/\s+/);
      words[words.length - 1] = cleanSuggestion;
      newText = words.join(" ");
    }

    setInputText(newText);
    handleTextChange(newText);

    const newPosition = cursorPosition + newText.length;
    setSelection({ start: newPosition, end: newPosition });

    if (inputRef.current) {
      inputRef.current.setNativeProps({
        text: newText,
        selection: { start: newPosition, end: newPosition },
      });
    }
    inputRef.current?.focus();
  };

  const handleWordMeaningSearch = () => {
    setIsKeyboardVisible(false);
    /*
    This function toggles the word meaning search mode.
    If the word meaning search mode is active, it disables the word meaning search.
    Otherwise, it enables the word meaning search.
    */

    setIsWordMeaningSearchActive(!isWordMeaningSearchActive);
    if (!isWordMeaningSearchActive) {
      disableCorrection();
    }
  };

  useEffect(() => {
    if (isWordMeaningSearchActive) {
      const words = inputText.split(" ");
      console.log(words);
      for (let i = 0; i < words.length; i++) {
        if (handleWordPress(words[i])) {
          break;
        }
      }
    }
  }, [isWordMeaningSearchActive]);

  const handleCorrection = async () => {
    setIsKeyboardVisible(false);
    if (inputText.trim() === "") {
      return;
    }
    /*
    This function is used to check the spelling of the input text and get correction suggestions.
    It sets the loadingCorrection state variable to true, creates the body object with the input text and settings,
    and sends a POST request to the API to get the correction suggestions.
    It then processes the response to get the correction suggestions and sets the correction data in the state variable.
    If there are correction suggestions, it sets the isCorrectionActive state variable to true.
    */
    setLoadingCorrection(true);
    const body = {
      sentence_yrl: inputText,
      ortography: topLanguage.orthography,
      selected_next_words: "",
      selected_dict_words: "",
      disable_dic: !settings.enableDictionary,
      disable_next: !settings.enableNextWordSuggestion,
      disable_word_meaning: !settings.enableWordMeaning,
      disable_data_collection: !settings.enableDataCollection,
    };

    disableWordMeaningSearch();

    try {
      const response = await axios.post(
        `${apiUrl}/correct_yrl`,
        body,
        axiosSpellConfig
      );

      if (response.data && response.data.corrected_sentence) {
        const originalWords = inputText.match(
          /[\p{L}]+|[-[\](){}!?.]|[\d]+|[\p{P}\s]+/gu
        );
        if (!originalWords) {
          return;
        }
        const correctionSuggestions = response.data.corrected_sentence;
        const corrections: Record<string, CorrectionInfo> = {};

        const filteredWords = originalWords.filter(
          (word) => word.trim() !== ""
        );
        filteredWords.forEach((word, index) => {
          // Verifica se há sugestões para esta palavra
          if (
            correctionSuggestions[index] &&
            correctionSuggestions[index].length > 0
          ) {
            const firstSuggestion = correctionSuggestions[index][0];

            // Só adiciona correções se a primeira sugestão for diferente da palavra original
            if (word !== firstSuggestion) {
              // Filtra as sugestões para remover a palavra original se ela existir
              const uniqueSuggestions = correctionSuggestions[index].filter(
                (suggestion) => suggestion !== word
              );

              const key = `${word}-${index}`;
              corrections[key] = {
                word,
                suggestion: firstSuggestion,
                position: index,
                allSuggestions: uniqueSuggestions,
              };
            }
          }
        });

        if (Object.keys(corrections).length > 0) {
          setCorrectionData(corrections);
          setIsCorrectionActive(true);
          // LAIZ: Correction of ortografy.

          // LAIZ:correctionsLog  as sugestões de correção
          // LAIZ:Para cada entrada davariavel corrections, usa a palavra como chave e a lista de sugestões como valor
          let correctionData: { word: string; alternatives: string[] }[] = [];
          Object.entries(corrections).forEach((item) => {
            const key = item[0].split("-")[0];
            const value = item[1]; // LAIZ: informações da correção
            // LAIZ:Armazena as sugestões usando a palavra como chave
            correctionData.push({
              word: key,
              alternatives: value,
            });
          });

          let correctionLog = {};
          // GI: Caso a coleta de dados esteja habilitada, salva o log de correção com os dados, caso contrário, salva um objeto vazio
          if (settings.enableDataCollection) {
            correctionLog = {
              language: topLanguage.name,
              ortography: topLanguage.orthography,
              sentence: inputText,
              timestamp: timestamp,
              corrections: correctionData,
            };
          } else {
            correctionLog = {
              language: null,
              ortography: null,
              sentence: null,
              timestamp: timestamp,
              corrections: null,
            };
          }

          addCorrectionLog(correctionLog);
          incrementLogCount();
          console.log("Log de correção salvo:", correctionLog);
          console.log("Logs de Correção:", correctionLogs.current);
        } else {
          console.log("Nenhuma correção sugerida para o texto digitado.");
          setIsKeyboardVisible(false);
          showFeedback(
            getText("correction.success.title"),
            getText("correction.success.message")
          );
          setCorrectionData({});
          setIsCorrectionActive(false);
          setSelectedCorrectionWord(null);
        }
        // GI: Salva o log de correçãoAdd commentMore actions
        if (settings.enableNextWordSuggestion) {
          let suggestionLog = {};
          // GI: Caso a coleta de dados esteja habilitada, salva o log de sugestão com os dados, caso contrário, salva um objeto vazio
          if (settings.enableDataCollection) {
            suggestionLog = {
              language: topLanguage.name,
              orthography: topLanguage.orthography,
              final_sentence: inputText,
              predictions: suggestionData,
            };
          } else {
            suggestionLog = {
              language: null,
              orthography: null,
              final_sentence: null,
              predictions: null,
            };
          }
          if (suggestionData.length > 0) {
            addSuggestionLog(suggestionLog);
            incrementLogCount();
          }
        }
      }
    } catch (error: any) {
      if (
        error.code === "ECONNABORTED" ||
        error.message?.includes("timeout") ||
        (error.message === "Network Error" && axiosSpellConfig.timeout <= 1000)
      ) {
        showError(getText("errors.correction.timeout"));
      } else {
        showError(getText("errors.correction.generic"));
      }
      setCorrectionData({});
      setIsCorrectionActive(false);
      setSelectedCorrectionWord(null);
    } finally {
      setLoadingCorrection(false);
    }
  };

  const handleSuggestionError = () => {
    setLastChar("");
  };

  const handleCorrectionPress = (word: string, position: number) => {
    setIsKeyboardVisible(false);
    /*
    This function receives the word and position as parameters and sets the selected correction word and position in the state variables.
    It then checks if there is a correction suggestion for the word at the specified position.
    If there is a correction suggestion, it sets the selected correction word and position in the state variables.
    Otherwise, it logs a message saying that there are no correction suggestions for the word at the specified position.
    */
    const correctionKey = `${word}-${position}`;
    if (correctionData[correctionKey]) {
      setSelectedCorrectionWord(word); // Armazena apenas a palavra, sem a posição
      setSelectedCorrectionPosition(position); // Armazena a posição separadamente
    } else {
    }
  };

  const handleCorrectionSuggestionPress = (suggestion: string) => {
    /*
    This function receives the suggestion as a parameter and applies the correction suggestion to the input text.
    It first checks if there is a selected correction word and position in the state variables.
    If there is a selected correction word and position, it gets the correction suggestion for the word and position.
    It then checks if the suggestion is "Manter como está".
    If the suggestion is "Manter como está", it deletes the correction suggestion.
    Otherwise, it applies the correction suggestion to the input text at the specified position.
    It then updates the correction data in the state variable and checks if there are any more corrections.
    If there are no more corrections, it disables the correction mode.
    Finally, it clears the selected correction word and position in the state variables.
    */
    if (!selectedCorrectionWord || selectedCorrectionPosition === null) return;

    const words = inputText.split(/\s+/);
    let updatedCorrectionData = { ...correctionData };

    // Busca a correção usando a chave completa (palavra + posição)
    const correctionKey = `${selectedCorrectionWord}-${selectedCorrectionPosition}`;
    const correction = correctionData[correctionKey];

    if (correction) {
      if (suggestion === correction.word) {
        // Se a sugestão for "Manter como está"
        delete updatedCorrectionData[correctionKey];
      } else {
        // Aplica a correção para a posição específica
        words[selectedCorrectionPosition] = suggestion;
        delete updatedCorrectionData[correctionKey];
      }
    }

    const newText = words.join(" ");
    setInputText(newText);

    setCorrectionData(updatedCorrectionData);

    // Se não houver mais correções, desative o modo de correção
    if (Object.keys(updatedCorrectionData).length === 0) {
      setIsCorrectionActive(false);
    }

    // Limpa a seleção atual
    setSelectedCorrectionWord(null);
    setSelectedCorrectionPosition(null);
  };

  const disableCorrection = () => {
    /*
    This function is used to disable the correction mode.
    It sets the correction data in the state variable to an empty object and sets the isCorrectionActive state variable to false.
    It also clears the selected correction word and position in the state variables
    */
    setIsCorrectionActive(false);
    setCorrectionData({});
  };

  // LAIZ: Função para incrementar o contador de logs
  const incrementLogCount = () => {
    setLogCount((prevCount) => {
      const newCount = prevCount + 1;
      if (newCount >= 15) {
        console.log("mostrando popup de log");
        setLogPopupVisible(true);
      }
      return newCount;
    });
  };

  const disableWordMeaningSearch = () => {
    /*
    This function is used to disable the word meaning search mode.
    It sets the word meanings in the state variable to an empty array and sets the isWordMeaningSearchActive state variable to false.
    It also clears the selected word in the state variable.
    */
    setIsWordMeaningSearchActive(false);
    setSelectedWord(null);
  };

  const handleCopy = (text: string) => {
    setIsKeyboardVisible(false);
    /*
    This function receives the text as a parameter and copies the text to the clipboard.
    It also shows a temporary pop-up message to inform the user that the text has been copied.
    */
    Clipboard.setString(text);
    setSelectedWord(null);
    disableWordMeaningSearch();
    disableCorrection();
    showTempPopUp(getText("copySuccess.title"), getText("copySuccess.message"));
  };

  const handlePaste = async () => {
    setIsKeyboardVisible(false);

    /*
    This function is used to paste text from the clipboard to the input text.
    It gets the text from the clipboard and adds it to the input text.
    It also clears the selected word, disables the word meaning search, and disables the correction mode
    */
    const text = await Clipboard.getString();
    setInputText((prevText) => prevText + text);
    setSelectedWord(null);
    disableWordMeaningSearch();
    disableCorrection();
  };

  const handleClear = (isInput: boolean) => {
    /*
    This function receives a boolean parameter isInput and clears the input or output text based on the value of the parameter.
    It also clears the selected word, disables the word meaning search, and disables the correction mode.
    */
    if (isInput) {
      setInputText("");
    } else {
      setOutputText("");
    }
    setSelectedWord(null);
    disableWordMeaningSearch();
    disableCorrection();
  };

  const handleWordPress = (word: string) => {
    /*
    This function receives the word as a parameter and searches for the word in the dictionary entries.
    It first converts the word to lowercase and searches for all variations of the word in the dictionary entries.
    If the word is found in the dictionary entries, it sets the selected word to the base word and sets the word meanings in the state variable.
    Otherwise, it logs a message saying that the word was not found in the dictionary.
    */
    if (isWordMeaningSearchActive) {
      const lowerCasedWord = word.toLowerCase();

      // Procura por todas as variações da palavra (incluindo numeradas)
      const allMeanings: string[] = [];

      // Verifica correspondência exata e variações numeradas
      Object.keys(dictionaryEntries.dic_words).forEach((key) => {
        const baseWord = key.replace(/\(\d+\)$/, "").toLowerCase();
        if (baseWord === lowerCasedWord) {
          allMeanings.push(...dictionaryEntries.dic_words[key]);
        }
      });

      if (allMeanings.length > 0) {
        setSelectedWord(lowerCasedWord); // Guarda apenas a palavra base
        // Você precisará adicionar um estado para guardar todos os significados
        setWordMeanings(allMeanings);
        return true;
      } else {
        return false;
      }
    }
  };

  const handleSettingsPress = () => {
    /*
    This function is used to show the settings modal.
    It sets the isSettingsModalVisible state variable to true
    */
    setIsSettingsModalVisible(true);
  };

  const handleSettingChange = (setting: string, value: boolean) => {
    /*
    This function receives the setting and value as parameters and sets the setting in the state variable.
    It is used to update the settings based on the user's preferences.
    */
    setSettings((prevSettings) => ({
      ...prevSettings,
      [setting]: value,
    }));
  };

  /*
  These functions are used to handle the input focus, blur, and press events.
  They are used to manage the input focus state and clear the selected word, disable the word meaning search, and disable the correction mode.
  */
  const handleInputFocus = () => {
    setIsKeyboardVisible(true);
    setIsInputFocused(true);
  };

  const handleInputBlur = () => {
    setIsKeyboardVisible(false);
    setIsInputFocused(false);
  };

  const handleInputPress = () => {
    setSelectedWord(null);
    disableCorrection();
    disableWordMeaningSearch();
  };

  const handleTranslate = async () => {
    setIsKeyboardVisible(false);
    /*
    This function is used to translate the input text to the target language.
    It first sets the loading state to true and disables the correction and word meaning search modes.
    In the loading state, the user cannot interact with the app.
    It then creates the body object with the input text and settings and sends a POST request to the API to translate the text.
    If the translation is successful, it sets the output text to the translated text.
    If the translation fails, it shows an error message to the user.
    Finally, it sets the loading state to false.
    */
    setLoading(true);
    disableCorrection();
    disableWordMeaningSearch();
    const commonSettings = {
      disable_dic: settings.enableDictionary,
      disable_next: settings.enableNextWordSuggestion,
      disable_word_meaning: settings.enableWordMeaning,
      disable_data_collection: settings.enableDataCollection,
    };

    let body;

    if (topLanguage.name === "Português") {
      body = bodyMapper["Português"]({
        sentence_pt: inputText,
        ortography: bottomLanguage.orthography,
        ...commonSettings,
      });
    } else if (topLanguage.name === "Nheengatu") {
      if (bottomLanguage.name === "Inglês") {
        body = bodyMapper["NheengatuEn"]({
          sentence_yrl: inputText,
          ortography: topLanguage.orthography,
          selected_next_words: "",
          selected_dict_words: "",
          ...commonSettings,
        });
      } else if (bottomLanguage.name === "Nheengatu") {
        body = bodyMapper["NheengatuConvert"]({
          sentence: inputText,
        });
      } else {
        body = bodyMapper["Nheengatu"]({
          sentence_yrl: inputText,
          ortography: topLanguage.orthography,
          selected_next_words: "",
          selected_dict_words: "",
          ...commonSettings,
        });
      }
    } else if (topLanguage.name === "Inglês") {
      body = bodyMapper["Inglês"]({
        sentence_en: inputText,
        ortography: topLanguage.orthography,
        ...commonSettings,
      });
    }

    try {
      const response = await axios.post(
        `${apiUrl}/${LanguageToUrlMapper[topLanguage.name][bottomLanguage.name]}`,
        body,
        axiosTranslateConfig
      );
      //ISA: Extract the translation result from the responseAdd commentMore actions
      const translationResult =
        response.data[
          LanguageToOutputFieldMapper[topLanguage.name][bottomLanguage.name]
        ];

      setOutputText(translationResult);

      let translationLog = {};

      if (settings.enableDataCollection) {
        translationLog = {
          timestamp: timestamp,
          source_sentence: inputText,
          target_sentence: translationResult,
          source_language: topLanguage.name,
          source_ortography: topLanguage.orthography,
          target_language: bottomLanguage.name,
          target_ortography: bottomLanguage.orthography,
        };
      } else {
        translationLog = {
          timestamp: timestamp,
          source_sentence: null,
          target_sentence: null,
          source_language: null,
          source_ortography: null,
          target_language: null,
          target_ortography: null,
        };
      }

      addTranslationLog(translationLog);
      incrementLogCount();
      if (settings.enableNextWordSuggestion) {
        let suggestionLog = {};
        // GI: Caso a coleta de dados esteja habilitada, salva o log de sugestão com os dados, caso contrário, salva um objeto vazio
        if (settings.enableDataCollection) {
          suggestionLog = {
            language: topLanguage.name,
            orthography: topLanguage.orthography,
            final_sentence: inputText,
            predictions: suggestionData,
          };
        } else {
          suggestionLog = {
            language: null,
            orthography: null,
            final_sentence: null,
            predictions: null,
          };
        }
        addSuggestionLog(suggestionLog);
        console.log("Log de sugestão salvo:", suggestionLog);
        console.log("Logs de Sugestão:", suggestionLogs.current);
        console.log("predictions", suggestionData);
      }
      inputRef.current?.blur();
    } catch (error) {
      if (
        error.code === "ECONNABORTED" ||
        error.message?.includes("timeout") ||
        error.message === "Network Error"
      ) {
        showError(getText("errors.translation.timeout"));
      } else {
        showError(getText("errors.translation.generic"));
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    getSpecialCharacters();
  }, [topLanguage, languages]);

  useEffect(() => {
    if (!isKeyboardVisible) {
      inputRef.current?.blur;
      setIsInputFocused(false);
    }
  }, [isKeyboardVisible]);

  const handleNavigateToFeedback = () => {
    setIsLoading(true);

    setFeedbackData({
      translationLogs: translationLogs.current,
      suggestionLogs: suggestionLogs.current,
      correctionLogs: correctionLogs.current,
      topLanguage: topLanguage.name,
      topOrthography: topLanguage.orthography,
      disableDic: !settings.enableDictionary,
      disableNext: !settings.enableNextWordSuggestion,
      disableWordMeaning: !settings.enableWordMeaning,
      dataCollection: settings.enableDataCollection,
      username: username,
    });

    navigation.navigate("FeedbackPage");

    setIsLoading(false);
  };

  // LAIZ:variavel para contar os logs gerados
  const [logCount, setLogCount] = useState(0);

  // LAIZ: variavel para controlar a exibição do popup de log
  const [logPopupVisible, setLogPopupVisible] = useState(false);

  // LAIZ: Função para resetar os logs de tradução, sugestão e correção
  const resetFeedbackLogs = () => {
    translationLogs.current = [];
    suggestionLogs.current = [];
    correctionLogs.current = [];
    setLogCount(0);
    console.log("Logs resetados.");
  };

  // Laiz: Verifica se foi passado o parâmetro resetFeedback para resetar os logs

  useEffect(() => {
    if (
      route.params !== undefined &&
      (route.params as any).resetFeedback === true
    ) {
      // LAIZ: Se o parâmetro resetFeedback for true reseta os logs

      console.log("ENTROU");
      console.log(logPopupVisible);
      setLogPopupVisible(false);

      resetFeedbackLogs();
      console.log(
        "zerou depois que veio do feedback",
        translationLogs,
        suggestionLogs,
        correctionLogs
      );
    }
  }, [route.params]);

  const tCount = translationLogs.current.length;
  const sCount = suggestionLogs.current.length;
  const cCount = correctionLogs.current.length;
  console.log("CONTAGEM");
  console.log(tCount);
  console.log(sCount);
  console.log(cCount);
  console.log(feedbackCountThreshold);
  console.log(feedbackTimeThreshold);

  useLogPopupHandler(
    tCount,
    sCount,
    cCount,
    username,
    () => setLogPopupVisible(true),
    logPopupVisible,
    feedbackCountThreshold,
    feedbackTimeThreshold * 1000
  );

  return (
    <SafeAreaView className="flex-1 bg-white">
      <StatusBar barStyle="light-content" backgroundColor="#0F62FE" />
      <View className="bg-[#0F62FE] py-2 px-3 flex-row justify-between items-center">
        <Text
          className={`text-white  font-semibold`}
          style={{ fontSize: responsiveFontSize(20) }}
        >
          {getText("appTitle")}
        </Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
          {/* Ícone de feedback */}
          {isFeedbackEnabled && (
            <TouchableOpacity onPress={handleNavigateToFeedback}>
              <Image
                source={feedbackIcon}
                style={{ width: 55, height: 55 }}
                resizeMode="contain"
              />
            </TouchableOpacity>
          )}

          {/* Ícone de configurações (continua usando SVGIcon) */}
          <TouchableOpacity onPress={handleSettingsPress}>
            <SVGIcon
              xml={settingsIconSvg}
              width={24}
              height={24}
              color="white"
            />
          </TouchableOpacity>
        </View>
      </View>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        className="flex-1"
      >
        <TouchableWithoutFeedback
          onPress={() => {
            inputRef.current?.blur();
            setIsWordMeaningSearchActive(false);
            Keyboard.dismiss();
          }}
        >
          <View className="flex-1 p-2 m-2 border-2 border-[#0F62FE] rounded-lg">
            <ScrollView scrollEnabled={!isInputFocused}>
              <View className="flex-row justify-between items-center">
                <LanguageSelector
                  languages={languages}
                  language={topLanguage}
                  setLanguage={setTopLanguage}
                  setIsKeyboardVisible={setIsKeyboardVisible}
                  handleLanguageChange={handleLanguageChange}
                  availableLanguages={
                    isIndigenousOnTop
                      ? indigenousLanguages
                      : targetLanguages.filter(
                          (language) => language !== "Nheengatu"
                        )
                  }
                  availableOrthographies={
                    isIndigenousOnTop && languages && topLanguage.name
                      ? Object.keys(
                          languages[topLanguage.name]["ortographies"]
                        ).filter(
                          (orthography) =>
                            orthography !== bottomLanguage.orthography
                        )
                      : undefined
                  }
                  uiLanguage={targetLanguage}
                />
                <View className="flex-row">
                  <ActionButtons
                    type="input"
                    onCopy={() => handleCopy(inputText)}
                    onPaste={handlePaste}
                    onClear={() => {
                      setIsKeyboardVisible(false);

                      return handleClear(true);
                    }}
                    onShare={() => {
                      setIsKeyboardVisible(false);
                      return Share.share({ message: inputText });
                    }}
                    language={targetLanguage}
                  />
                </View>
              </View>
              <TranslationBox
                placeholder={getText("inputPlaceholder")}
                editable={true}
                value={inputText}
                onChangeText={handleTextChange}
                inputRef={inputRef}
                autoCorrect={false}
                autoCompleteType="off"
                isInput={true}
                onFocus={handleInputFocus}
                onBlur={handleInputBlur}
                enableWordMeaningSearch={
                  isWordMeaningSearchActive && isIndigenousOnTop
                }
                dictionaryData={dictionaryEntries.dic_words}
                onWordPress={handleWordPress}
                enableCorrection={isIndigenousOnTop && isCorrectionActive}
                correctionData={correctionData}
                onCorrectionPress={handleCorrectionPress}
                onInputPress={handleInputPress}
                selection={selection}
                onSelectionChange={handleSelectionChange}
              />
              <View className="flex-row justify-between mt-2 mb-6">
                <TouchableOpacity
                  className="bg-[#0F62FE] px-2 py-2 rounded mr-1 flex-[1.2]"
                  onPress={handleCorrection}
                  disabled={
                    !isIndigenousOnTop ||
                    loadingCorrection ||
                    !canCorrrect() ||
                    inputText.trim() === ""
                  }
                  style={{
                    backgroundColor:
                      !isIndigenousOnTop ||
                      loadingCorrection ||
                      !canCorrrect() ||
                      inputText.trim() === ""
                        ? "grey"
                        : "#4d5358",
                    opacity:
                      !isIndigenousOnTop ||
                      loadingCorrection ||
                      !canCorrrect() ||
                      inputText.trim() === ""
                        ? 0.5
                        : 1,
                  }}
                >
                  <Text
                    className={`text-white font-medium text-center`}
                    style={{ fontSize: responsiveFontSize(12) }}
                    numberOfLines={1}
                  >
                    {loadingCorrection
                      ? getText("checkingButton")
                      : getText("checkButton")}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  className="bg-[#0F62FE] px-2 py-2 rounded mx-1 flex-[1.2]"
                  onPress={handleWordMeaningSearch}
                  disabled={
                    !isIndigenousOnTop ||
                    !haveDictionary() ||
                    inputText.trim() === "" ||
                    isWordMeaningSearchActive
                  }
                  style={{
                    backgroundColor:
                      !isIndigenousOnTop ||
                      !haveDictionary() ||
                      inputText.trim() === "" ||
                      isWordMeaningSearchActive
                        ? "grey"
                        : "#4d5358",
                    opacity:
                      !isIndigenousOnTop ||
                      !haveDictionary() ||
                      inputText.trim() === "" ||
                      isWordMeaningSearchActive
                        ? 0.5
                        : 1,
                  }}
                >
                  <Text
                    className={`text-white  font-medium text-center`}
                    numberOfLines={1}
                    style={{ fontSize: responsiveFontSize(12) }}
                  >
                    {getText("examineButton")}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  className="bg-[#0F62FE] px-2 py-2 rounded ml-1 flex-[0.8]"
                  onPress={handleTranslate}
                  disabled={
                    loading || !canTranslate() || inputText.trim() === ""
                  }
                  style={{
                    backgroundColor:
                      loading || !canTranslate() || inputText.trim() === ""
                        ? "grey"
                        : "#4d5358",
                    opacity:
                      loading || !canTranslate() || inputText.trim() === ""
                        ? 0.5
                        : 1,
                  }}
                >
                  <Text
                    className={`text-white  font-medium text-center`}
                    numberOfLines={1}
                    style={{ fontSize: responsiveFontSize(12) }}
                  >
                    {isBothLanguagesIndigenous
                      ? loading
                        ? getText("convertingButton")
                        : getText("convertButton")
                      : loading
                        ? getText("translatingButton")
                        : getText("translateButton")}
                  </Text>
                </TouchableOpacity>
              </View>

              <View className="flex-row items-center my-2">
                <View className="flex-1 h-px bg-[#C1C7CD]" />
                <TouchableOpacity
                  onPress={toggleLanguages}
                  className="bg-[#0F62FE] rounded-full p-2 mx-2"
                >
                  <SVGIcon
                    xml={refreshCwIconSvg}
                    width={24}
                    height={24}
                    color="white"
                  />
                </TouchableOpacity>
                <View className="flex-1 h-px bg-[#C1C7CD]" />
              </View>
              <View className="flex-row justify-between items-center">
                <LanguageSelector
                  languages={languages}
                  language={bottomLanguage}
                  setIsKeyboardVisible={setIsKeyboardVisible}
                  setLanguage={setBottomLanguage}
                  availableLanguages={
                    isIndigenousOnTop ? targetLanguages : indigenousLanguages
                  }
                  availableOrthographies={
                    languages &&
                    bottomLanguage.name &&
                    languages[bottomLanguage.name] &&
                    languages[bottomLanguage.name]["ortographies"]
                      ? Object.keys(
                          languages[bottomLanguage.name]["ortographies"]
                        ).filter(
                          (orthography) =>
                            orthography !== topLanguage.orthography
                        )
                      : []
                  }
                  uiLanguage={targetLanguage}
                />
                <ActionButtons
                  type="output"
                  onCopy={() => handleCopy(outputText)}
                  onClear={() => handleClear(false)}
                  onShare={() => Share.share({ message: outputText })}
                  language={targetLanguage}
                />
              </View>
              <Text
                className={` text-gray-600 mt-3 mb-0.5`}
                style={{ fontSize: responsiveFontSize(12) }}
              >
                {getText("translatedText")}
              </Text>
              <TranslationBox
                placeholder={getText("translationPlaceholder")}
                editable={false}
                value={outputText}
                autoCorrect={false}
                autoCompleteType="off"
                isInput={false}
              />
            </ScrollView>
            {/* LAIZ:  Blue Button feedback */}
          </View>
        </TouchableWithoutFeedback>

        {isWordMeaningSearchActive && (
          <WordMeaningBox
            word={selectedWord}
            meanings={wordMeanings}
            onClose={() => {
              setSelectedWord(null);
              setWordMeanings([]);
            }}
            isVisible={!!selectedWord}
          />
        )}
        {selectedCorrectionWord && isCorrectionActive && (
          <CorrectionSuggestionBox
            word={selectedCorrectionWord}
            suggestions={
              Object.entries(correctionData).find(
                ([key, info]) => info.word === selectedCorrectionWord
              )?.[1]?.allSuggestions || []
            }
            onClose={() => setSelectedCorrectionWord(null)}
            onSuggestionPress={handleCorrectionSuggestionPress}
          />
        )}
        {isKeyboardVisible && isIndigenousOnTop && (
          <TouchableWithoutFeedback onPress={() => {}}>
            <View>
              <SuggestionBox
                isIndigenousLanguage={isIndigenousOnTop}
                currentWord={currentWord}
                lastChar={lastChar}
                onSuggestionPress={handleSuggestionPress}
                settings={settings}
                dictionaryData={dictionaryEntries.dic_words}
                dictionaryDataEn={dictionaryEntriesEn.dic_words}
                bottomLanguage={bottomLanguage.name}
                apiURL={apiUrl}
                inputText={inputText}
                showError={showError}
                onError={handleSuggestionError}
                uiLanguage={targetLanguage}
                orthography={topLanguage.orthography}
                canSuggestNextWords={canSuggestNextWord}
                haveDictionary={haveDictionary}
              />
            </View>
          </TouchableWithoutFeedback>
        )}
        {isKeyboardVisible && isIndigenousOnTop && (
          <SpecialCharacters
            characters={chars}
            onCharacterPress={insertSpecialCharacter}
            uiLanguage={targetLanguage}
          />
        )}
        <SettingsModal
          isVisible={isSettingsModalVisible}
          onClose={() => setIsSettingsModalVisible(false)}
          settings={settings}
          onSettingChange={handleSettingChange}
          language={targetLanguage}
        />
        <NoInternetPopup visible={!isConnected} />
        {!isConnected && (
          <View
            className="absolute inset-0 bg-transparent z-50"
            pointerEvents="box-none"
          />
        )}
        {/* LAIZ: Renderiza o popup quando atingir 15 logs e pergunta se quer fazer um feedback*/}
        {logPopupVisible && (
          <LogLimitPopup
            visible={logPopupVisible}
            title="Forneça um feedback da sua experiência"
            message="Você gostaria de fazer um feedback para melhorar a qualidade dos dados?"
            onClose={() => {
              // LAIZ: Se o usuário clicar em "Não", zera os logs
              resetFeedbackLogs();
              setLogPopupVisible(false);
            }}
            onPressSim={() => {
              handleNavigateToFeedback();
            }}
          />
        )}
        <FeedbackModal
          visible={feedbackModal.visible}
          title={feedbackModal.title}
          message={feedbackModal.message}
          type={feedbackModal.type}
          isTempPopUp={feedbackModal.isTempPopUp}
          onClose={() => {
            setIsKeyboardVisible(false);
            setIsInputFocused(false);
            inputRef.current?.blur();
            setFeedbackModal((prev) => ({ ...prev, visible: false }));
          }}
        />

        {isFeedbackEnabled && (
          <DataCapturePopup
            visible={dataCaptureVisible}
            onClose={handleCloseDataCapture}
          />
        )}
        <LoadingOverlay visible={loading || loadingCorrection} />
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

export default MainPage;
