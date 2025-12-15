// src/screens/MainPage/handlers/useLogPopupHandler.ts
import { useEffect } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useFeedback } from "../../../contexts/FeedbackContext";

export function useLogPopupHandler(
  translationCount: number = 0,
  suggestionCount: number = 0,
  correctionCount: number = 0,
  username: string,
  showPopup: () => void,
  isShowPopup: boolean,
  threshold: number = 5,
  cooldownMs: number = 12 * 60 * 60 * 1000
) {
  const storageKey = `lastLogPopup_${username}`;

  const { isFeedbackEnabled } = useFeedback();

  useEffect(() => {
    console.log("ENTROU 2");
    if (!isFeedbackEnabled) {
      return;
    }
    const total = translationCount + suggestionCount + correctionCount;
    console.log("TOTAL DE FEEDBACK: " + total);
    if (total < threshold) return;

    (async () => {
      try {
        const lastStr = await AsyncStorage.getItem(storageKey);
        const lastTs = lastStr ? parseInt(lastStr, 10) : 0;
        const now = Date.now();

        console.log("InformaçÕes: " + now, lastTs, cooldownMs);

        if (now - lastTs >= cooldownMs) {
          if (!isShowPopup) showPopup();
          await AsyncStorage.setItem(storageKey, now.toString());
        }
      } catch (error) {
        console.error("[useLogPopupHandler] AsyncStorage error:", error);
        showPopup();
      }
    })();
  }, [
    translationCount,
    suggestionCount,
    correctionCount,
    username,
    showPopup,
    threshold,
    cooldownMs,
    isFeedbackEnabled,
  ]);
}
