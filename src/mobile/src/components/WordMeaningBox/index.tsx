import React from "react";
import { View, Text, ScrollView, TouchableOpacity } from "react-native";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- WordMeaningBox is a component that displays the meaning of a word.
- It is used to display the meaning of a word in a box.
- The component receives props with the word, the list of meanings, and a function to close the box.
*/

interface WordMeaningBoxProps {
  word: string | null;
  meanings: string[];
  onClose: () => void;
  isVisible: boolean;
}

const WordMeaningBox: React.FC<WordMeaningBoxProps> = ({
  word,
  meanings,
  onClose,
  isVisible,
}) => {
  if (!isVisible) return null;

  return (
    <View className="h-1/2 bg-[#E8F1FF] border-3 border-[#0F62FE]">
      <TouchableOpacity onPress={onClose} className="bg-[#0F62FE] p-1">
        <Text className="text-white font-bold text-base">Significado</Text>
      </TouchableOpacity>

      <ScrollView
        className="flex-1 p-2"
        keyboardShouldPersistTaps="always"
        persistentScrollbar={true}
      >
        <View>
          <Text>
            <Text
              className={`font-bold`}
              style={{ fontSize: responsiveFontSize(16) }}
            >
              {word} :{" "}
            </Text>
            <Text style={{ fontSize: responsiveFontSize(14) }}>
              [{meanings.join(" | ")}]
            </Text>
          </Text>
        </View>
      </ScrollView>
    </View>
  );
};

export default WordMeaningBox;
