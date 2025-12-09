import React, {
    createContext,
    Dispatch,
    SetStateAction,
    useState
} from "react";

type LoadingContextType = {
    isLoading: boolean;
    setIsLoading: Dispatch<SetStateAction<boolean>>;
};

const defaultContext: LoadingContextType = {
    isLoading: false,
    setIsLoading: () => {},
};

export const LoadingContext =
    createContext<LoadingContextType>(defaultContext);

export const LoadingProvider = ({
    children
}: {
    children: React.ReactNode;
}) => {
    const [isLoading, setIsLoading] = useState<boolean>(false
    );

    return (
        <LoadingContext.Provider
            value={{
                isLoading,
                setIsLoading,
            }}
        >
            {children}
        </LoadingContext.Provider>
    );
};
