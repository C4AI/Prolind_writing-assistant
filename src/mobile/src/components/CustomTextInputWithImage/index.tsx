import React from "react";
import {
  View,
  TextInput,
  TouchableOpacity,
  TextInputProps,
  Image,
} from "react-native";

/*
COMPONENT DESCRIPTION:
- CustomTextInputWithImage is a component that displays a text input field with an image.
- It is used to create a text input field with a placeholder, value, and image.
- The component receives props to define the placeholder, value, and function to handle text changes.
- The component also receives a prop to define if the text input should be hidden (password field).
*/

interface CustomTextInputWithImageProps extends TextInputProps {
  iconSource: any; // Tipo para a fonte da imagem
  onToggleVisibility?: () => void;
  accessibilityLabel?: string;
  secureTextEntry?: boolean;
  font: string;
}

const CustomTextInputWithImage: React.FC<CustomTextInputWithImageProps> = ({
  iconSource,
  onToggleVisibility,
  accessibilityLabel,
  secureTextEntry,
  font,
  ...props
}) => {
  return (
    <View className="p-0 w-full mb-4">
      <View className="flex-row items-center border border-gray-300 rounded-lg">
        <TextInput
          {...props}
          secureTextEntry={secureTextEntry}
          accessible={true}
          accessibilityLabel={accessibilityLabel}
          className="flex-1 p-4"
          placeholderTextColor="#787878"
          autoCapitalize="none"
        />
        {onToggleVisibility && (
          <TouchableOpacity
            onPress={onToggleVisibility}
            accessible={true}
            accessibilityLabel={
              secureTextEntry ? "Mostrar senha" : "Esconder senha"
            }
            accessibilityRole="button"
            style={{ marginRight: 10 }}
          >
            <Image source={iconSource} style={{ width: 20, height: 20 }} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

export default CustomTextInputWithImage;
