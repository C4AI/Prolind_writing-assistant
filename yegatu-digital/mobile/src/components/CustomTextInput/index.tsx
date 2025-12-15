import React from "react";
import { TextInput, View } from "react-native";

/*
COMPONENT DESCRIPTION:
- CustomTextInput is a component that displays a text input field.
- It is used to create a text input field with a placeholder and value.
- The component receives props to define the placeholder, value, and function to handle text changes.
- The component also receives a prop to define if the text input should be hidden (password field).
*/

interface CustomTextInputProps {
  placeholder: string;
  value: string;
  onChangeText: (text: string) => void;
  secureTextEntry?: boolean;
  accessibilityLabel?: string;
  font: string;
}

const CustomTextInput: React.FC<CustomTextInputProps> = ({
  placeholder,
  value,
  onChangeText,
  secureTextEntry,
  accessibilityLabel,
  font,
}) => {
  return (
    <View className="p-0 w-full mb-4">
      <TextInput
        className="border border-gray-300 rounded-lg p-4 w-full"
        placeholder={placeholder}
        value={value}
        onChangeText={onChangeText}
        secureTextEntry={secureTextEntry}
        accessible={true}
        accessibilityLabel={accessibilityLabel}
        placeholderTextColor="#787878"
        autoCapitalize="none"
      />
    </View>
  );
};

export default CustomTextInput;
