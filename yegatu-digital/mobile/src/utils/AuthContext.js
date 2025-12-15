import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(null);
  const [tokenEn, setTokenEn] = useState(null); 

  return (
    <AuthContext.Provider value={{ token, setToken ,tokenEn,setTokenEn}}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  return useContext(AuthContext);
};
