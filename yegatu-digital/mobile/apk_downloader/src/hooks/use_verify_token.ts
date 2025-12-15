import { isAxiosError } from "axios";
import { Dispatch, SetStateAction, useEffect } from "react";
import { axiosInstance } from "../constants/axios_instance";
import { useLoading } from "./use_loading";
import { toast } from "react-toastify";

export const useVerifyToken = (
    setMessage?: Dispatch<SetStateAction<string>>
) => {
    const { setIsLoading } = useLoading();

    useEffect(() => {
        // verify the token every time the page is loaded and redirects to the login page if the token is invalid
        const handleVerifyToken = async () => {
            setIsLoading(true);
            try {
                const route =
                    window.location.href.includes("/en") ||
                    window.location.href.includes("_en")
                        ? "/verify_token_en"
                        : "/verify_token";
                await axiosInstance.get(route);
            } catch (err) {
                if (isAxiosError(err)) {
                    toast.error(err.message);
                }
            }
            setIsLoading(false);
        };
        handleVerifyToken();
    }, [setMessage, setIsLoading]);
};
