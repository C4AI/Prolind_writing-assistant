import axios from "axios";
import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TouchableWithoutFeedback,
  ActivityIndicator,
} from "react-native";
import { useAuth } from "../../utils/AuthContext";
import { responsiveFontSize } from "../../utils/FontContext";
// GI: export suggestionLog to be used in the parent component
export { suggestionData };

/*
COMPONENT DESCRIPTION:
- SuggestionBox is a component that displays a list of suggestions for the current word being typed.
- It is used to display suggestions for the current word in the input text.
- The component receives props with the current word, the last character typed, the list of suggestions, and functions to handle the selection of a suggestion.
- The component also receives props with the settings to enable or disable the dictionary, next word suggestion, and word meaning.
- The component receives props with the dictionary data and the language of the user interface.
*/

let suggestionData: { input_sentence: string; alternatives: string[] }[] = [];

interface SuggestionBoxProps {
  isIndigenousLanguage: boolean;
  currentWord: string;
  lastChar: string;
  onSuggestionPress: (suggestion: string) => void;
  settings: {
    enableDictionary: boolean;
    enableNextWordSuggestion: boolean;
    enableWordMeaning: boolean;
  };
  dictionaryData: { [key: string]: string[] };
  dictionaryDataEn: { [key: string]: string[] };
  bottomLanguage: string;
  apiURL: string;
  inputText: string;
  showError: (message: string) => void;
  onError: () => void;
  uiLanguage: string;
  orthography: string;
  canSuggestNextWords: () => boolean;
  haveDictionary: () => boolean;
}

const uiText = {
  pt: {
    suggestions: "Sugestões",
    dictionary: "Dicionário",
    suggestionError: "A busca por sugestões excedeu o tempo limite.",
  },
  en: {
    suggestions: "Suggestions",
    dictionary: "Dictionary",
    suggestionError: "The search for suggestions exceeded the time limit.",
  },
};

const SuggestionBox: React.FC<SuggestionBoxProps> = ({
  isIndigenousLanguage,
  currentWord,
  lastChar,
  onSuggestionPress,
  settings,
  dictionaryData,
  dictionaryDataEn,
  bottomLanguage,
  apiURL,
  inputText,
  showError,
  onError,
  uiLanguage,
  orthography,
  canSuggestNextWords,
  haveDictionary,
}) => {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [cachedResults, setCachedResults] = useState<{
    [key: string]: string[];
  }>({});
  const [loading, setLoading] = useState<boolean>(false);
  const { token } = useAuth();
  const [nextWordSuggestions, setNextWordSuggestions] = useState<string[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const axiosConfig = {
    timeout: 60000,
    headers: {
      Authorization: token,
      "Content-Type": "application/json",
      accept: "application/json",
    },
  };

  const filterSuggestions = (word: string): string[] => {
    const cacheKey = `${bottomLanguage}_${word}`;

    if (cachedResults[cacheKey]) {
      return cachedResults[cacheKey];
    }

    const lowerCaseWord = word.toLowerCase();
    const result = Object.keys(dictionaryData)
      .filter((dictWord) => dictWord.startsWith(lowerCaseWord))
      .slice(0, 20);
    setCachedResults((prevCache) => ({
      ...prevCache,
      [cacheKey]: result,
    }));

    return result;
  };

  const getText = (key: string): string => {
    const lang = uiLanguage === "Inglês" ? "en" : "pt";
    return uiText[lang][key];
  };

  const getNextWordUrls = async () => {
    /*
    - This function is responsible for fetching the URLs to get the next word suggestions.
    - It sends a request to the API to get the URLs for the next word suggestions.
    - The URLs are then used to fetch the next word suggestions based on the input text.
    - If an error occurs during the process, it displays an error message.
    */
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      setLoading(true);
      const result = await axios.get(
        apiURL + `/next_word_info?orthography=${orthography}`,
        {
          ...axiosConfig,
          signal: abortController.signal,
        }
      );
      console.log(result.data);

      const nextWordFetchUrl = result.data.next_word_fetch_url + "/fetch";
      const nextWordUrl = result.data.next_word_url + "/api";
      const listOfWordsToExclude = result.data.list_of_words_to_exclude;
      const filter = result.data.filter;

      const nextWordSuggestionsResponse = await axios.post(
        nextWordFetchUrl,
        {
          input: inputText,
          url: nextWordUrl,
          orthography: orthography.toLowerCase(),
        },
        {
          ...axiosConfig,
          signal: abortController.signal,
        }
      );

      if (!abortController.signal.aborted) {
        console.log(nextWordSuggestionsResponse.data);

        const nextWords = nextWordSuggestionsResponse.data.map(
          (word: string) => {
            if (listOfWordsToExclude.includes(word.toLowerCase().trim())) {
              if (filter === "remove_word") {
                return "";
              } else if (filter === "remove_sentence") {
                return "";
              } else if (filter === "do_nothing") {
                return word;
              } else if (filter === "redact_word") {
                return "*******";
              } else {
                return word;
              }
            } else {
              return word;
            }
          }
        );
        console.log(nextWords);
        const nextWordsFiltered = nextWords.filter(
          (word: string) => word.trim() !== ""
        );
        setNextWordSuggestions(nextWordsFiltered);
        setSuggestions(nextWordsFiltered);
        // GI: store the suggestions in the global variable
        if (inputText.trim().length > 0) {
          const chave = inputText.slice(0, -1);
          const alternatives = nextWordSuggestionsResponse.data;

          // Adiciona sugestão ao Array de sugestões
          suggestionData.push({
            input_sentence: chave,
            alternatives: alternatives,
          });
        }
      }
    } catch (error: any) {
      if (!axios.isCancel(error)) {
        if (
          error.code === "ECONNABORTED" ||
          error.message?.includes("timeout") ||
          error.message === "Network Error"
        ) {
          showError(getText("suggestionError"));
        }
        onError();
      }
    } finally {
      if (!abortController.signal.aborted) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    if (
      !isIndigenousLanguage ||
      (!settings.enableDictionary && !settings.enableNextWordSuggestion)
    ) {
      setSuggestions([]);
      return;
    }

    if (inputText.trim().length === 0 && settings.enableNextWordSuggestion) {
      suggestionData = [];
    }

    if (lastChar === " " || /[.,!?]/.test(lastChar)) {
      setSuggestions([]);
      if (!canSuggestNextWords()) {
        return;
      }
      getNextWordUrls();
    } else if (currentWord && settings.enableDictionary) {
      if (!haveDictionary()) {
        return;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setLoading(false);
      const filteredSuggestions = filterSuggestions(currentWord);
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [
    isIndigenousLanguage,
    currentWord,
    lastChar,
    dictionaryData,
    dictionaryDataEn,
    settings,
    bottomLanguage,
  ]);

  if (!isIndigenousLanguage || (suggestions.length === 0 && !loading))
    return null;

  return (
    <TouchableWithoutFeedback onPress={() => {}}>
      <View className="h-36 bg-[#F2F4F8] border-3 border-[#0F62FE]">
        <View className="bg-[#0F62FE] p-1">
          <Text
            className={`text-white font-bold `}
            style={{ fontSize: responsiveFontSize(16) }}
          >
            {lastChar === " " || /[.,!?]/.test(lastChar)
              ? getText("suggestions")
              : getText("dictionary")}
          </Text>
        </View>
        {loading ? (
          <View className="flex-1 justify-center items-center">
            <ActivityIndicator size="large" color="#0F62FE" />
          </View>
        ) : (
          <ScrollView
            className="flex-1"
            keyboardShouldPersistTaps="always"
            persistentScrollbar={true}
          >
            {suggestions.map((suggestion, index) => (
              <TouchableOpacity
                key={index}
                className="p-2 border-b border-[#C1C7CD] flex flex-row"
                onPress={() => onSuggestionPress(suggestion)}
              >
                <Text
                  className={` font-black`}
                  style={{ fontSize: responsiveFontSize(12) }}
                >
                  {suggestion}
                </Text>
                <Text style={{ fontSize: responsiveFontSize(12) }}>
                  {settings.enableWordMeaning &&
                    (bottomLanguage === "Inglês"
                      ? dictionaryDataEn[suggestion] &&
                        `: ${dictionaryDataEn[suggestion].join(", ")}`
                      : dictionaryData[suggestion] &&
                        `: ${dictionaryData[suggestion].join(", ")}`)}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </View>
    </TouchableWithoutFeedback>
  );
};

export default SuggestionBox;
