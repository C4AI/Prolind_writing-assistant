import React from "react";
import { View, TouchableOpacity, Text } from "react-native";
import SVGIcon from "../SVGIcon";

import copyIcon from "../../assets/copy";
import filePlusIcon from "../../assets/file-plus";
import trashIcon from "../../assets/trash-2";
import shareIcon from "../../assets/share-icon";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- ActionButtons is a component that displays a row of buttons with icons and text.
- It is used to copy, paste, clear, and share text.
- The component receives props to define the type of buttons to display (input or output) and the actions to be performed when the buttons are pressed.
- The component also receives the language to display the text in the correct language.
*/

interface ActionButtonsProps {
  type: "input" | "output";
  onCopy: () => void;
  onPaste?: () => void;
  onClear: () => void;
  onShare?: () => void;
  language: string;
}

const uiText = {
  pt: {
    copy: "Copiar",
    paste: "Colar",
    clear: "Limpar",
    share: "Compartilhar",
  },
  en: {
    copy: "Copy",
    paste: "Paste",
    clear: "Clear",
    share: "Share",
  },
};

const ActionButtons: React.FC<ActionButtonsProps> = ({
  type,
  onCopy,
  onPaste,
  onClear,
  onShare,
  language,
}) => {
  const getText = (key: string): string => {
    const lang = language === "Inglês" ? "en" : "pt";
    return uiText[lang][key];
  };

  return (
    <View className="flex-row">
      <TouchableOpacity className="items-center mr-3" onPress={onCopy}>
        <SVGIcon xml={copyIcon} width={20} height={20} color="black" />
        <Text className={` mt-1`} style={{ fontSize: responsiveFontSize(10) }}>
          {getText("copy")}
        </Text>
      </TouchableOpacity>
      {type === "input" && (
        <>
          <TouchableOpacity className="items-center mr-3" onPress={onPaste}>
            <SVGIcon xml={filePlusIcon} width={20} height={20} color="black" />
            <Text
              className={` mt-1`}
              style={{ fontSize: responsiveFontSize(10) }}
            >
              {getText("paste")}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity className="items-center" onPress={onClear}>
            <SVGIcon xml={trashIcon} width={20} height={20} color="black" />
            <Text
              className={` mt-1`}
              style={{ fontSize: responsiveFontSize(10) }}
            >
              {getText("clear")}
            </Text>
          </TouchableOpacity>
        </>
      )}
      <>
        <TouchableOpacity className="items-center ml-3" onPress={onShare}>
          <SVGIcon xml={shareIcon} width={20} height={20} color="black" />
          <Text
            className={` mt-1`}
            style={{ fontSize: responsiveFontSize(10) }}
          >
            {getText("share")}
          </Text>
        </TouchableOpacity>
      </>
    </View>
  );
};

export default ActionButtons;
