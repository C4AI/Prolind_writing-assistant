import { useEffect } from "react";

export const useResetLocalStorage = () => {
    useEffect(() => {
        localStorage.removeItem("isDictionaryEnabled");
        localStorage.removeItem("isNextWordEnabled");
        localStorage.removeItem("isMeaningEnabled");
        localStorage.removeItem("token");
        localStorage.removeItem("left_language");
        localStorage.removeItem("left_orthography");
        localStorage.removeItem("right_language");
        localStorage.removeItem("right_orthography");
    }, []);
};
