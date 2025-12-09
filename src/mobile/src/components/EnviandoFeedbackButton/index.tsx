// components/EnviandoFeedbackButton.tsx
import React from "react";
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ViewStyle,
  TextStyle,
} from "react-native";

interface EnviandoFeedbackButtonProps {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  accessibilityLabel?: string;
}

const EnviandoFeedbackButton: React.FC<EnviandoFeedbackButtonProps> = ({
  title,
  onPress,
  disabled = false,
  style,
  textStyle,
  accessibilityLabel,
}) => {
  return (
    <TouchableOpacity
      style={[
        styles.button,
        disabled ? styles.buttonDisabled : styles.buttonEnabled,
        style,
      ]}
      onPress={onPress}
      disabled={disabled}
      accessible
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
    >
      <Text style={[styles.text, textStyle]} numberOfLines={1}>
        {title}
      </Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    paddingVertical: 20,
    paddingHorizontal: 0,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonEnabled: {
    backgroundColor: "#0F62FE",
  },
  buttonDisabled: {
    backgroundColor: "#A0AEC0",
  },
  text: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
  },
});

export default EnviandoFeedbackButton;
