import { AuthProvider } from "./providers/AuthProvider";
import { LoadingProvider } from "./providers/LoadingProvider";
import { ToastContainer } from "react-toastify";
import AppRoutes from "./routes/AppRoutes";

function App() {
  return (
    <AuthProvider>
      <LoadingProvider>
        <ToastContainer
          limit={5}
          position="top-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
        <AppRoutes />
      </LoadingProvider>
    </AuthProvider>
  );
}

export default App;
