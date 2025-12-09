import React, { createContext, useEffect, useMemo, useState } from "react";

// provider that stores the token of the user and loads it in the localstorage

type AuthContextType = {
    token: string | null;
    setToken: React.Dispatch<React.SetStateAction<string | null>>;
};

const defaultContext: AuthContextType = {
    token: "",
    setToken: () => {}
};

export const AuthContext = createContext<AuthContextType>(defaultContext);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [token, setToken] = useState(localStorage.getItem("token"));

    // if the token exists it loads it in the localstorage and if it doesn't it removes it (every time the token changes)
    useEffect(() => {
        if (token) {
            localStorage.setItem("token", token);
        } else {
            localStorage.removeItem("token");
        }
    }, [token]);

    const contextValue = useMemo(() => ({ token, setToken }), [token]);

    return (
        <AuthContext.Provider value={contextValue}>
            {children}
        </AuthContext.Provider>
    );
};
