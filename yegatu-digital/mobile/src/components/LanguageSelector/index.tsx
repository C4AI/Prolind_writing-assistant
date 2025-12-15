import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  FlatList,
  Pressable,
} from "react-native";
import SVGIcon from "../SVGIcon";
import chevronDownIcon from "../../assets/chevron-down";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- LanguageSelector is a component that displays a dropdown to select the language and orthography.
- It is used to select the language and orthography for the input text.
- The component receives props with the selected language and orthography, the available languages and orthographies, and functions to handle the selection of a language or orthography.
- The component also receives a prop with the language of the user interface to display the text in the correct language.
*/

interface LanguageProps {
  name: string;
  orthography?: string;
}

interface LanguageSelectorProps {
  language: LanguageProps;
  setLanguage: React.Dispatch<React.SetStateAction<LanguageProps>>;
  showTranslateButton?: boolean;
  availableLanguages: string[];
  availableOrthographies?: string[];
  uiLanguage: string;
  languages?: never;
  handleLanguageChange?: (
    selectedLanguage: string,
    selectedOrthography: string
  ) => Promise<void>;
  setIsKeyboardVisible: React.Dispatch<React.SetStateAction<boolean>>;
}

const uiText = {
  pt: {
    language: "Idioma",
    orthography: "Ortografia",
  },
  en: {
    language: "Language",
    orthography: "Orthography",
  },
};

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  language,
  setLanguage,
  showTranslateButton,
  availableLanguages,
  availableOrthographies,
  uiLanguage,
  topLanguageOrthography,
  languages,
  handleLanguageChange,
  setIsKeyboardVisible,
}) => {
  const [showLanguageModal, setShowLanguageModal] = useState(false);
  const [showOrthographyModal, setShowOrthographyModal] = useState(false);

  const renderDropdownItem = (
    item: string,
    onSelect: (item: string) => void,
    isSelected: boolean
  ) => (
    <TouchableOpacity
      onPress={() => onSelect(item)}
      className={`p-3 border-b border-[#C1C7CD] ${isSelected ? "bg-[#C1C7CD]" : ""}`}
    >
      <Text
        className={`${isSelected ? "font-semibold text-black" : "text-[#000000]"}`}
      >
        {item}
      </Text>
    </TouchableOpacity>
  );

  const getText = (key: string): string => {
    const lang = uiLanguage === "Inglês" ? "en" : "pt";
    return uiText[lang][key];
  };

  const ModalContent = ({
    title,
    data,
    onSelect,
    currentSelection,
    closeModal,
    isOrthography = false,
  }) => (
    <View
      className="bg-white rounded-t-lg w-80"
      style={{ maxHeight: 300, minHeight: 200 }}
    >
      <View className="bg-[#0F62FE] py-2 px-3">
        <Text
          className={`text-white font-semibold`}
          style={{ fontSize: responsiveFontSize(14) }}
        >
          {title}
        </Text>
      </View>
      <FlatList
        data={data}
        renderItem={({ item }) => {
          const sufix = isOrthography
            ? item === "CLG"
              ? "(nheengatu)"
              : "(yeẽgatu)"
            : "";
          const text = item + " " + sufix;
          return renderDropdownItem(
            text,
            (selected) => {
              onSelect(selected.split(" ")[0]);
              closeModal();
            },
            item === currentSelection
          );
        }}
        keyExtractor={(item) => item}
        showsVerticalScrollIndicator={true}
        scrollIndicatorInsets={{ right: 1 }}
      />
    </View>
  );

  return (
    <View className="mt-1">
      <View className="flex-row justify-between items-center">
        <TouchableOpacity
          onPress={() => {
            setIsKeyboardVisible(false);
            return setShowLanguageModal(true);
          }}
          className="flex-row items-center"
        >
          <Text
            className={` font-semibold`}
            style={{ fontSize: responsiveFontSize(14) }}
          >
            {language.name === "Inglês" ? "English" : language.name}
          </Text>
          <SVGIcon xml={chevronDownIcon} width={16} height={16} color="black" />
        </TouchableOpacity>
      </View>
      {language.orthography && (
        <TouchableOpacity
          onPress={() => {
            setIsKeyboardVisible(false);
            return setShowOrthographyModal(true);
          }}
          className="flex-row items-center mt-1"
        >
          <Text
            className={` text-gray-500`}
            style={{ fontSize: responsiveFontSize(10) }}
          >
            {getText("orthography")}: {language.orthography}
          </Text>
          <SVGIcon
            xml={chevronDownIcon}
            width={14}
            height={14}
            color="#7F8A8E"
          />
        </TouchableOpacity>
      )}

      <Modal visible={showLanguageModal} transparent animationType="fade">
        <Pressable
          className="flex-1 justify-center items-center"
          style={{ backgroundColor: "rgba(0,0,0,0.8)" }}
          onPress={() => setShowLanguageModal(false)}
        >
          <ModalContent
            title={getText("language")}
            data={availableLanguages}
            onSelect={(selectedLanguage) => {
              let orthography = "";
              if (languages) {
                orthography = Object.keys(
                  languages[selectedLanguage]["ortographies"]
                )[0];
              }
              setLanguage({
                name: selectedLanguage,
                orthography: orthography,
              });
              if (handleLanguageChange)
                handleLanguageChange(selectedLanguage, orthography);
            }}
            currentSelection={language.name}
            closeModal={() => setShowLanguageModal(false)}
          />
        </Pressable>
      </Modal>

      <Modal visible={showOrthographyModal} transparent animationType="fade">
        <Pressable
          className="flex-1 justify-center items-center"
          style={{ backgroundColor: "rgba(0,0,0,0.8)" }}
          onPress={() => setShowOrthographyModal(false)}
        >
          <ModalContent
            title={getText("orthography")}
            isOrthography={true}
            data={availableOrthographies || []}
            onSelect={(selectedOrthography) => {
              if (handleLanguageChange) {
                handleLanguageChange(language.name, selectedOrthography);
                return setLanguage((prev) => ({
                  ...prev,
                  orthography: selectedOrthography,
                }));
              }
            }}
            currentSelection={language.orthography}
            closeModal={() => setShowOrthographyModal(false)}
          />
        </Pressable>
      </Modal>
    </View>
  );
};

export default LanguageSelector;
