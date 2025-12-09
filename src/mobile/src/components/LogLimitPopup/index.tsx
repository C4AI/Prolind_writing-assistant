// src/components/LogLimitPopup.tsx
import React from "react";
import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";

interface LogLimitPopupProps {
  visible: boolean;
  title: string;
  message: string;
  onClose: () => void;
  onPressSim: () => void;
}

const LogLimitPopup: React.FC<LogLimitPopupProps> = ({
  visible,
  title,
  message,
  onClose,
  onPressSim,
}) => (
  <Modal visible={visible} transparent animationType="fade">
    <View style={styles.overlay}>
      <View style={styles.container}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.message}>{message}</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.button, styles.noButton]}
            onPress={onClose}
          >
            <Text style={styles.buttonText}>Não</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.yesButton]}
            onPress={onPressSim}
          >
            <Text style={styles.buttonText}>Sim</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  </Modal>
);

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)", // fundo semitransparente
    justifyContent: "center",
    alignItems: "center",
  },
  container: {
    width: "80%",
    maxWidth: 300,
    backgroundColor: "#ffffff",
    borderRadius: 8,
    padding: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: "700",
    marginBottom: 12,
    color: "#000000",
    textAlign: "center",
  },
  message: {
    fontSize: 16,
    marginBottom: 16,
    color: "#333333",
    textAlign: "center",
  },
  buttonRow: {
    flexDirection: "row",
    borderRadius: 6,
    overflow: "hidden",
  },
  button: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
  },
  noButton: {
    backgroundColor: "#6b7280", // cinza escuro
  },
  yesButton: {
    backgroundColor: "#0f62fe", // azul IBM
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "500",
  },
});

export default LogLimitPopup;
