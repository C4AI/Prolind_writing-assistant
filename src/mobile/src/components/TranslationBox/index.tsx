import React, { useState, useEffect } from "react";
import {
  View,
  TextInput,
  TouchableOpacity,
  Text,
  NativeSyntheticEvent,
  TextInputContentSizeChangeEventData,
  ScrollView,
  TextInputSelectionChangeEventData,
} from "react-native";

/*
COMPONENT DESCRIPTION:
- TranslationBox is a component that displays a text input field with clickable words.
- It is the main component on the app
- The component receives props to define the placeholder, value, and function to handle text changes.
- The component also receives props to define if the text is editable, and if it should display clickable words.
- The component also receives props with the dictionary data and the correction data.
*/

interface TranslationBoxProps {
  placeholder: string;
  editable?: boolean;
  value?: string;
  onChangeText?: (text: string) => void;
  inputRef?: React.RefObject<TextInput>;
  autoCorrect?: boolean;
  autoCompleteType?:
    | "off"
    | "none"
    | "text"
    | "password"
    | "email"
    | "name"
    | "tel"
    | "username"
    | "password-new"
    | "postal-code"
    | "street-address"
    | "cc-number"
    | "cc-csc"
    | "cc-exp"
    | "cc-exp-month"
    | "cc-exp-year";
  isInput: boolean;
  onFocus?: () => void;
  onBlur?: () => void;
  enableWordMeaningSearch?: boolean;
  dictionaryData?: Record<string, string>;
  onWordPress?: (word: string) => void;
  enableCorrection?: boolean;
  correctionData?: Record<string, string[]>;
  onCorrectionPress?: (word: string, position: number) => void;
  onInputPress?: () => void;
  selection?: { start: number; end: number };
  onSelectionChange?: (
    e: NativeSyntheticEvent<TextInputSelectionChangeEventData>
  ) => void;
}

const TranslationBox: React.FC<TranslationBoxProps> = ({
  placeholder,
  editable = true,
  value,
  onChangeText,
  inputRef,
  autoCorrect = false,
  autoCompleteType = "off",
  isInput,
  onFocus,
  onBlur,
  enableWordMeaningSearch,
  dictionaryData = {},
  onWordPress,
  enableCorrection,
  correctionData = {},
  onCorrectionPress,
  onInputPress,
  selection,
  onSelectionChange,
}) => {
  const [fontSize, setFontSize] = useState(22);
  const [contentHeight, setContentHeight] = useState(36);
  const minFontSize = 16;
  const maxFontSize = 22;
  const maxHeight = 144;

  useEffect(() => {
    if (value) {
      const newSize = Math.max(
        maxFontSize - Math.floor(value.length / 15),
        minFontSize
      );
      setFontSize(newSize);
    } else {
      setFontSize(maxFontSize);
    }
  }, [value]);

  const onContentSizeChange = (
    event: NativeSyntheticEvent<TextInputContentSizeChangeEventData>
  ) => {
    const newHeight = Math.min(event.nativeEvent.contentSize.height, maxHeight);
    setContentHeight(Math.max(newHeight, 36));
  };

  const renderClickableText = (text: string) => {
    /*
    This function receives a text and returns a list of Text components, with each word being clickable.
    The function should return a list of Text components, with each word being clickable.
    The function should also highlight the words that have a dictionary entry or a correction suggestion.
    If the word has a dictionary entry, the background color should be '#be95ff'.
    If the word has a correction suggestion, the background color should be '#ff8389'.
    The function should call the onWordPress function when a word with a dictionary entry is pressed.
    The function should call the onCorrectionPress function when a word with a correction suggestion is pressed.
    */
    const words = text.split(/\s+/);
    return (
      <View style={{ flexDirection: "row", flexWrap: "wrap" }}>
        {words.map((word, index) => {
          if (word.trim() === "") {
            return (
              <Text
                key={`space-${index}`}
                style={{ marginRight: 5, fontSize: fontSize }}
              >
                {" "}
              </Text>
            );
          }

          const lowerCasedWord = word.toLowerCase();
          const correctionInfo =
            enableCorrection &&
            Object.values(correctionData).find(
              (info) => info.word === word && info.position === index
            );
          const hasDictionaryEntry =
            enableWordMeaningSearch &&
            Object.keys(dictionaryData).some(
              (key) =>
                key.toLowerCase().startsWith(`${lowerCasedWord}(`) ||
                key.toLowerCase() === lowerCasedWord
            );

          const isClickable = correctionInfo || hasDictionaryEntry;

          let backgroundColor = "transparent";
          if (correctionInfo) {
            backgroundColor = "#ff8389";
          } else if (hasDictionaryEntry) {
            backgroundColor = "#be95ff";
          }

          return (
            <TouchableOpacity
              key={`${word}-${index}-${!!correctionInfo}-${hasDictionaryEntry}`}
              onPress={() => {
                if (correctionInfo) {
                  onCorrectionPress(word, index); // Usa o índice diretamente para identificar a posição da palavra
                } else if (hasDictionaryEntry) {
                  onWordPress?.(word);
                }
              }}
              disabled={!isClickable}
            >
              <Text
                style={{
                  color: isInput ? "black" : "text-gray-500",
                  marginRight: 5,
                  fontSize: fontSize,
                  backgroundColor: backgroundColor,
                  padding: 2,
                  borderRadius: 3,
                }}
              >
                {word}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    );
  };

  const handleContainerPress = () => {
    if (isInput) {
      onInputPress?.();
      if (enableWordMeaningSearch || enableCorrection) {
        onChangeText?.(value || "");
      }
      inputRef?.current?.focus();
    }
  };

  const handleTextInputPress = () => {
    onInputPress?.();
  };

  return (
    <TouchableOpacity
      className={`mt-2 rounded relative ${isInput ? "h-36 bg-gray-50" : "bg-gray-200"}`}
      style={!isInput ? { minHeight: contentHeight } : {}}
      onPress={handleContainerPress}
      activeOpacity={1}
    >
      {isInput && !enableWordMeaningSearch && !enableCorrection ? (
        <TextInput
          className={`p-2 ${isInput ? "text-black" : "text-gray-500"}`}
          style={{
            fontSize: fontSize,
            paddingBottom: 8,
            height: isInput ? "100%" : contentHeight,
          }}
          placeholder={placeholder}
          multiline
          editable={editable}
          value={value}
          onChangeText={onChangeText}
          ref={inputRef}
          placeholderTextColor="#C1C7CD"
          autoCorrect={autoCorrect}
          autoCompleteType={autoCompleteType}
          spellCheck={false}
          contextMenuHidden={true}
          keyboardType="visible-password"
          textAlignVertical="top"
          onContentSizeChange={onContentSizeChange}
          onFocus={onFocus}
          onBlur={onBlur}
          onPressIn={handleTextInputPress}
          selection={selection}
          onSelectionChange={onSelectionChange}
        />
      ) : (
        <ScrollView style={{ padding: 8 }}>
          {renderClickableText(value || "")}
        </ScrollView>
      )}
    </TouchableOpacity>
  );
};

export default TranslationBox;
