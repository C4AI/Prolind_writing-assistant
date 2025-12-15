import axios from "axios";
import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
} from "react";
import { useAuth } from "../utils/AuthContext";

interface FeedbackContextData {
  translationLogs: any[];
  suggestionLogs: any[];
  correctionLogs: any[];
  topLanguage: string;
  topOrthography?: string;
  disableDic: boolean;
  disableNext: boolean;
  disableWordMeaning: boolean;
  dataCollection: boolean;
  username: string;
  setFeedbackData: (data: Partial<FeedbackContextData>) => void;
  clearFeedbackData: () => void;
  isFeedbackEnabled: boolean;
  feedbackCountThreshold: number;
  feedbackTimeThreshold: number;
  setIsFeedbackEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  setFeedbackTimeThreshold: React.Dispatch<React.SetStateAction<number>>;
  setFeedbackCountThreshold: React.Dispatch<React.SetStateAction<number>>;
}

const FeedbackContext = createContext<FeedbackContextData | undefined>(
  undefined
);

export const FeedbackProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [translationLogs, setTranslationLogs] = useState<any[]>([]);
  const [suggestionLogs, setSuggestionLogs] = useState<any[]>([]);
  const [correctionLogs, setCorrectionLogs] = useState<any[]>([]);
  const [topLanguage, setTopLanguage] = useState<string>("");
  const [topOrthography, setTopOrthography] = useState<string | undefined>(
    undefined
  );
  const [disableDic, setDisableDic] = useState<boolean>(false);
  const [disableNext, setDisableNext] = useState<boolean>(false);
  const [disableWordMeaning, setDisableWordMeaning] = useState<boolean>(false);
  const [dataCollection, setDataCollection] = useState<boolean>(false);
  const [username, setUsername] = useState<string>("");
  const [isFeedbackEnabled, setIsFeedbackEnabled] = useState(false);
  const [feedbackCountThreshold, setFeedbackCountThreshold] = useState(5);
  const [feedbackTimeThreshold, setFeedbackTimeThreshold] = useState(0);

  const setFeedbackData = (data: Partial<FeedbackContextData>) => {
    if (data.translationLogs !== undefined)
      setTranslationLogs(data.translationLogs);
    if (data.suggestionLogs !== undefined)
      setSuggestionLogs(data.suggestionLogs);
    if (data.correctionLogs !== undefined)
      setCorrectionLogs(data.correctionLogs);
    if (data.topLanguage !== undefined) setTopLanguage(data.topLanguage);
    if (data.topOrthography !== undefined)
      setTopOrthography(data.topOrthography);
    if (data.disableDic !== undefined) setDisableDic(data.disableDic);
    if (data.disableNext !== undefined) setDisableNext(data.disableNext);
    if (data.disableWordMeaning !== undefined)
      setDisableWordMeaning(data.disableWordMeaning);
    if (data.dataCollection !== undefined)
      setDataCollection(data.dataCollection);
    if (data.username !== undefined) setUsername(data.username);
  };

  const clearFeedbackData = () => {
    setTranslationLogs([]);
    setSuggestionLogs([]);
    setCorrectionLogs([]);
    setTopLanguage("");
    setTopOrthography(undefined);
    setDisableDic(false);
    setDisableNext(false);
    setDisableWordMeaning(false);
    setDataCollection(false);
  };

  const { token, setToken } = useAuth();
  const headerS = {
    Authorization: token,
    "Content-Type": "application/json",
    accept: "application/json",
  };

  return (
    <FeedbackContext.Provider
      value={{
        translationLogs,
        suggestionLogs,
        correctionLogs,
        topLanguage,
        topOrthography,
        disableDic,
        disableNext,
        disableWordMeaning,
        dataCollection,
        username,
        setFeedbackData,
        clearFeedbackData,
        isFeedbackEnabled,
        feedbackCountThreshold,
        feedbackTimeThreshold,
        setFeedbackCountThreshold,
        setFeedbackTimeThreshold,
        setIsFeedbackEnabled,
      }}
    >
      {children}
    </FeedbackContext.Provider>
  );
};

export const useFeedback = (): FeedbackContextData => {
  const context = useContext(FeedbackContext);
  if (!context)
    throw new Error("useFeedback must be used within FeedbackProvider");
  return context;
};
