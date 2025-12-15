import React from 'react';
import { TouchableOpacity, Text, Image, View } from 'react-native';
import SVGIcon from '../SVGIcon';
import arrowRightIcon from '../../assets/arrow-right';

/*
COMPONENT DESCRIPTION:
- LoginButton is a component that displays a button with an icon and text.
- It is used to create a button with a title and an icon.
- The component receives props to define the title, icon, and action to be performed when the button is pressed.
- The component also receives props to define the color, style, and accessibility label of the button.
*/

interface LoginButtonProps {
  title: string;
  onPress: () => void;
  additionalStyle?: string;
  iconSource?: any;
  accessibilityLabel?: string;
  disabled?: boolean;
  color?: string;
}

const LoginButton: React.FC<LoginButtonProps> = ({
  title,
  onPress,
  additionalStyle,
  iconSource,
  accessibilityLabel,
  disabled,
  color,
}) => {
  return (
    <TouchableOpacity
      className={`flex-row items-center justify-between ${color} p-4 rounded ${additionalStyle}`}
      onPress={onPress}
      accessible={true}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
      disabled={disabled}
    >
      <Text className="w-[70%] text-white text-lg">{title}</Text>
      {iconSource ? (
        <Image source={iconSource} className="w-6 h-2 tint-white" />
      ) : (
        <SVGIcon xml={arrowRightIcon} width={24} height={24} color="white" />
      )}
    </TouchableOpacity>
  );
};

export default LoginButton;