import React from 'react';
import { View, Text, Modal, ActivityIndicator } from 'react-native';

/*
COMPONENT DESCRIPTION:
- NoInternetPopup is a component that displays a popup when there is no internet connection.
- It is used to inform the user that there is no internet connection and to check the connection.
- The component receives a prop to define if the popup should be visible.
*/

interface NoInternetPopupProps {
  visible: boolean;
}

const NoInternetPopup: React.FC<NoInternetPopupProps> = ({ visible }) => {
  return (
    <Modal
      transparent
      animationType="fade"
      visible={visible}
    >
      <View className="flex-1 justify-center items-center bg-black bg-opacity-50">
        <View className="bg-white p-5 rounded-lg items-center">
          <Text className="text-lg font-bold mb-2">Sem conexão com a internet</Text>
          <Text className="text-center mb-4">
            Por favor, verifique sua conexão e tente novamente.
          </Text>
          <ActivityIndicator size="large" color="#0F62FE" />
        </View>
      </View>
    </Modal>
  );
};

export default NoInternetPopup;