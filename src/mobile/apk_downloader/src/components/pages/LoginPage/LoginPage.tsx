import { useRef, useState } from "react";
import axios, { isAxiosError } from "axios";
import Spinner from "../../common/Spinner/Spinner";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../../hooks/use_auth";
import Input from "./utils/Input";
import Title from "../../common/Title/Title";
import Button from "../../common/Button/Button";
import ErrorMessage from "../../common/ErrorMessage/ErrorMessage";
import { useResetLocalStorage } from "../../../hooks/use_reset_local_storage";
import { toast } from "react-toastify";

export default function LoginPage() {
  const { setToken } = useAuth();
  const navigate = useNavigate();
  useResetLocalStorage();

  const usernameInputRef = useRef<HTMLInputElement>(null);
  const passwordInputRef = useRef<HTMLInputElement>(null);

  const [message, setMessage] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setMessage("");
    setIsLoading(true);

    if (usernameInputRef.current && passwordInputRef.current) {
      const username = usernameInputRef.current.value;
      const password = passwordInputRef.current.value;
      try {
        const uninterceptedAxiosInstance = axios.create({
          baseURL: import.meta.env.VITE_API_URL,
        });
        const response = await uninterceptedAxiosInstance.post("/auth", {
          username,
          password,
          //app_id: 2
        });
        setToken(response.data.token);
        navigate("/download");
      } catch (err) {
        if (isAxiosError(err)) {
          toast.error(err.message);
        }
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-start">
      {isLoading && (
        <div className="w-full h-full absolute opacity-50 bg-zinc-600 flex items-center justify-center">
          <Spinner />
        </div>
      )}
      <Title />
      <form
        onSubmit={handleLogin}
        className="bg-white rounded-lg shadow-md p-6 w-2/3  mb-auto lg:w-1/2 flex flex-col items-center border border-gray-300 gap-3"
      >
        <h2 className="text-xl lg:text-2xl font-bold mb-4 mr-auto">Login</h2>
        <Input placeholder="Usuário" type="text" ref={usernameInputRef} />
        <Input placeholder="Senha" type="password" ref={passwordInputRef} />
        <Button
          title="Entrar"
          style="bg-blue-500 text-white mt-4 py-2 px-8 rounded hover:bg-blue-700"
        />
      </form>
      <ErrorMessage message={message} />
    </div>
  );
}
