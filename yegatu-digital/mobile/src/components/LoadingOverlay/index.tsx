import React from 'react';
import { View, ActivityIndicator, StyleSheet, Modal } from 'react-native';

/*
COMPONENT DESCRIPTION:
- LoadingOverlay is a component that displays a loading overlay with a spinner.
- It is used to indicate that the application is processing a request.
- The component receives a prop to define if the overlay should be visible.
*/

const LoadingOverlay = ({ visible }: { visible: boolean }) => {
  return (
    <Modal transparent={true} animationType="none" visible={visible}>
      <View style={styles.overlay}>
        <ActivityIndicator size="large" color="#fff" />
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
});

export default LoadingOverlay;
