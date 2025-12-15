import React from "react";
import { View, Text, ScrollView, TouchableOpacity } from "react-native";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- CorrectionSuggestionBox is a component that displays a list of suggestions for correcting a word.
- It is used to display suggestions for correcting a word in the input text.
- The component receives props with the word to be corrected, the list of suggestions, and functions to handle the selection of a suggestion or closing the suggestion box.
*/

interface CorrectionSuggestionBoxProps {
  word: string;
  suggestions: string[];
  onClose: () => void;
  onSuggestionPress: (suggestion: string) => void;
}

const CorrectionSuggestionBox: React.FC<CorrectionSuggestionBoxProps> = ({
  word,
  suggestions,
  onClose,
  onSuggestionPress,
}) => {
  return (
    <View className="h-36 bg-[#FFCCCB] border-3 border-[#FF8389] pb-2">
      <TouchableOpacity onPress={onClose} className="bg-[#FF8389] p-1">
        <Text
          className={`text-white font-bold`}
          style={{ fontSize: responsiveFontSize(16) }}
        >
          Sugestões de Correção
        </Text>
      </TouchableOpacity>

      <ScrollView
        persistentScrollbar={true}
        className="flex-1 p-2"
        keyboardShouldPersistTaps="always"
      >
        <TouchableOpacity
          onPress={() => onSuggestionPress(word)}
          className="mb-2"
        >
          <Text
            className={` font-bold p-2 bg-white/90 rounded`}
            style={{ fontSize: responsiveFontSize(14) }}
          >
            {word} <Text className="text-gray-500">(Manter como está)</Text>
          </Text>
        </TouchableOpacity>

        {suggestions.map((suggestion, index) => (
          <TouchableOpacity
            key={index}
            onPress={() => onSuggestionPress(suggestion)}
            className="mb-2"
          >
            <Text
              className={` p-2 bg-white rounded font-bold`}
              style={{ fontSize: responsiveFontSize(14) }}
            >
              {suggestion}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};

export default CorrectionSuggestionBox;
