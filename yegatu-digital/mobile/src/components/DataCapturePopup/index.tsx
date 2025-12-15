import React from "react";
import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";

interface DataCapturePopupProps {
  visible: boolean;
  onClose: () => void;
}

const DataCapturePopup: React.FC<DataCapturePopupProps> = ({
  visible,
  onClose,
}) => (
  <Modal visible={visible} transparent animationType="fade">
    <View style={styles.overlay}>
      <View style={styles.container}>
        <Text style={styles.message}>
          <Text style={styles.bold}>
            A Captura dos dados dos campos tradutores{"\n"}
          </Text>
          está habilitada. Se deseja{" "}
          <Text style={styles.bold}>desabilitar</Text> essa funcionalidade,
          acesse as <Text style={styles.bold}>configurações</Text>.{" "}
          <Text style={styles.gear}>⚙️</Text>
        </Text>
        <TouchableOpacity style={styles.button} onPress={onClose}>
          <Text style={styles.buttonText}>Entendido</Text>
        </TouchableOpacity>
      </View>
    </View>
  </Modal>
);

export default DataCapturePopup;

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: "center", // mudou para centralizar
    alignItems: "center", // opcional, reforça centralização horizontal
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 20, // garante margem nas laterais
  },
  container: {
    backgroundColor: "#fff",
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 12,
    // sombras (iOS)
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    // sombra (Android)
    elevation: 5,
    width: "100%", // ocupa até as paddings do overlay
    maxWidth: 400, // limita largura em telas grandes
  },
  message: {
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
    color: "#333",
    marginBottom: 16,
  },
  bold: {
    fontWeight: "600",
  },
  gear: {
    fontSize: 16,
  },
  button: {
    backgroundColor: "#4B5563",
    paddingVertical: 12,
    borderRadius: 6,
  },
  buttonText: {
    color: "#fff",
    textAlign: "center",
    fontWeight: "500",
    fontSize: 16,
  },
});
