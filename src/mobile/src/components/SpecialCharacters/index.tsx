import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- SpecialCharacters is a component that displays a list of special characters.
- It is used to display a list of special characters that can be inserted into the input text.
- The component receives props with the list of special characters and a function to handle the insertion of a character.
*/

interface SpecialCharactersProps {
  characters: string[];
  onCharacterPress: (char: string) => void;
  uiLanguage: string;
}

const uiText = {
  pt: {
    specialCharacters: "Caracteres Especiais",
  },
  en: {
    specialCharacters: "Special Characters",
  },
};

const SpecialCharacters: React.FC<SpecialCharactersProps> = ({
  characters,
  onCharacterPress,
  uiLanguage,
}) => {
  const getText = (key: string): string => {
    const lang = uiLanguage === "Inglês" ? "en" : "pt";
    return uiText[lang][key];
  };

  return (
    <View className="bg-white p-2 border-t border-gray-300">
      <View className="flex-row justify-between">
        {characters.map((char, index) => (
          <TouchableOpacity
            key={index}
            className="w-9 h-9 border border-[#0F62FE] bg-[#0F62FE] rounded-md items-center justify-center"
            onPress={() => {
              onCharacterPress(char);
            }}
          >
            <Text
              className={` font-semibold text-white`}
              style={{ fontSize: responsiveFontSize(14) }}
            >
              {char}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
};

export default SpecialCharacters;
