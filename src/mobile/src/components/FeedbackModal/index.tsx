import React, { useEffect } from "react";
import { View, Text, Modal, TouchableOpacity } from "react-native";
import { responsiveFontSize } from "../../utils/FontContext";

/*
COMPONENT DESCRIPTION:
- FeedbackModal is a component that displays a modal with feedback messages.
- It is used to display messages to the user about the result of an action.
- The component receives props with the message, title, and type of feedback to be displayed.
- The component also receives a prop to define if the modal should be closed automatically after a certain time.
*/

interface FeedbackModalProps {
  visible: boolean;
  title: string;
  message: string;
  onClose: () => void;
  type?: "error" | "success" | "info" | "warning";
  isTempPopUp?: boolean;
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({
  visible,
  title,
  message,
  onClose,
  type = "info",
  isTempPopUp = false,
}) => {
  useEffect(() => {
    if (visible && isTempPopUp) {
      const timer = setTimeout(() => {
        onClose();
      }, 1200);

      return () => clearTimeout(timer);
    }
  }, [visible, isTempPopUp, onClose]);

  const getColors = () => {
    switch (type) {
      case "error":
        return {
          title: "text-red-600",
          button: "bg-[#DC2626]",
        };
      case "success":
        return {
          title: "text-green-600",
          button: "bg-[#059669]",
        };
      case "warning":
        return {
          title: "text-yellow-600",
          button: "bg-[#D97706]",
        };
      default:
        return {
          title: "text-[#0F62FE]",
          button: "bg-[#0F62FE]",
        };
    }
  };

  const colors = getColors();

  return (
    <Modal transparent visible={visible} animationType="fade">
      <View className="flex-1 justify-center items-center bg-black/50">
        <View className="bg-white p-4 rounded-lg w-4/5 max-w-sm">
          <Text
            className={`font-bold mb-4 ${colors.title}`}
            style={{ fontSize: responsiveFontSize(18) }}
          >
            {title}
          </Text>
          <Text
            className={` mb-4`}
            style={{ fontSize: responsiveFontSize(16) }}
          >
            {message}
          </Text>
          {!isTempPopUp && (
            <TouchableOpacity
              onPress={onClose}
              className={`${colors.button} py-2 px-4 rounded self-end`}
            >
              <Text className="text-white font-medium">Fechar</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </Modal>
  );
};

export default FeedbackModal;
