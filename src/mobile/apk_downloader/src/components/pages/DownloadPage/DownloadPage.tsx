import { useEffect, useState } from "react";
import axios from "axios";
import Spinner from "../../common/Spinner/Spinner";
import Button from "../../common/Button/Button";
import Title from "../../common/Title/Title";
import { toast } from "react-toastify";
import Helper from "./utils/Helper";
import {
  Instruction,
  othersInstructions,
  samsungInstructions,
} from "./utils/instructions";

export default function DownloadPage() {
  const [version, setVersion] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [instructions, setInstructions] = useState<Instruction[]>([]);

  useEffect(() => {
    const loadVersion = async () => {
      try {
        const response = await axios.get("/version.txt");
        const version = Number(response.data);
        setVersion(version);
      } catch (err) {
        toast.error("Erro ao carregar o arquivo version.txt");
      }
      setIsLoading(false);
    };

    loadVersion();
  }, []);

  const handleDownload = () => {
    if (!version) return;

    const link = document.createElement("a");
    link.href = `/ALI-v${version}.apk`;
    link.download = `ALI-v${version}.apk`;
    link.click();
  };

  return (
    <div className="bg-gray-100 flex flex-col items-center justify-start min-h-screen flex-grow">
      {isLoading && (
        <div className="w-full h-full absolute opacity-50 bg-zinc-600 flex items-center justify-center">
          <Spinner />
        </div>
      )}
      <Title />
      <div className="bg-white rounded-lg shadow-md p-6 w-2/3 lg:w-1/2 flex items-center border border-gray-300 gap-3">
        <h2 className="text-sm md:text-lg lg:text-xl font-bold mr-auto">
          Versão do Aplicativo: {version}
        </h2>
        {version && (
          <Button
            title="Baixar"
            style="bg-blue-500 text-sm md:text-lg lg:text-xl text-white mt-4 py-2 px-4 md:px-8 rounded hover:bg-blue-700"
            onClick={() => handleDownload()}
          />
        )}
      </div>
      <div className="grow px-2 pt-4">
        {instructions.length ? (
          <Helper
            instructions={instructions}
            setInstructions={setInstructions}
          />
        ) : (
          <div className="text-center">
            <h2 className="text-sm md:text-lg lg:text-xl font-bold ">
              Instruções de instalação
            </h2>
            <span className="text-sm md:text-lg lg:text-xl ">
              Selecione o dispositivo que você possui
            </span>
            <ul className="">
              <li className="text-sm md:text-lg lg:text-xl flex items-center justify-center flex-col">
                <Button
                  title="Samsung"
                  style="bg-blue-500 text-sm md:text-lg lg:text-xl text-white mt-4 py-2 px-4 md:px-8 rounded hover:bg-blue-700"
                  onClick={() => {
                    setInstructions(samsungInstructions);
                  }}
                />
                <Button
                  title="Outro"
                  style="bg-blue-500 text-sm md:text-lg lg:text-xl text-white mt-4 py-2 px-4 md:px-8 rounded hover:bg-blue-700"
                  onClick={() => {
                    setInstructions(othersInstructions);
                  }}
                />
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
