import { useContext } from "react";
import { LoadingContext } from "../providers/LoadingProvider";

export const useLoading = () => {
    return useContext(LoadingContext);
};
