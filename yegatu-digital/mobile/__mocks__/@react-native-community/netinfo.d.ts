declare module '@react-native-community/netinfo' {
    export function addEventListener(listener: (state: { isConnected: boolean }) => void): () => void;
    export function fetch(): Promise<{ isConnected: boolean }>;
    export function useNetInfo(): { isConnected: boolean };
  }